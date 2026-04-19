//! Google Generative AI (Gemini) provider.
//!
//! Speaks the native Gemini REST shape (`POST :generateContent` and
//! `POST :streamGenerateContent?alt=sse`) so claw can be pointed at
//! `https://generativelanguage.googleapis.com/v1beta` directly OR at any
//! third-party proxy that exposes the same surface (LiteLLM, Vertex AI
//! gateways, internal company gateways like core-ai-platform.ikameglobal.com).
//!
//! Auth uses the `x-goog-api-key` header that the official Google API
//! recognises. Env vars: `GEMINI_API_KEY` (primary) or `GOOGLE_API_KEY`
//! (fallback — Google's own SDKs use this name). Base URL override:
//! `GEMINI_BASE_URL`.
//!
//! Wire-format docs: <https://ai.google.dev/api/rest/v1beta/models/generateContent>

#![allow(clippy::cast_possible_truncation)]

use std::collections::VecDeque;
use std::time::Duration;

use serde::Deserialize;
use serde_json::{json, Map, Value};

use crate::error::ApiError;
use crate::http_client::build_http_client_or_default;
use crate::types::{
    ContentBlockDelta, ContentBlockDeltaEvent, ContentBlockStartEvent, ContentBlockStopEvent,
    InputContentBlock, InputMessage, MessageDelta, MessageDeltaEvent, MessageRequest,
    MessageResponse, MessageStartEvent, MessageStopEvent, OutputContentBlock, StreamEvent,
    ToolChoice, ToolDefinition, ToolResultContentBlock, Usage,
};

use super::{preflight_message_request, Provider, ProviderFuture};

pub const DEFAULT_BASE_URL: &str = "https://generativelanguage.googleapis.com/v1beta";
const PROVIDER_NAME: &str = "Google Generative AI";
const ENV_VARS: &[&str] = &["GEMINI_API_KEY", "GOOGLE_API_KEY"];
const DEFAULT_INITIAL_BACKOFF: Duration = Duration::from_secs(1);
const DEFAULT_MAX_BACKOFF: Duration = Duration::from_secs(64);
const DEFAULT_MAX_RETRIES: u32 = 6;
const REQUEST_ID_HEADER: &str = "x-request-id";
const ALT_REQUEST_ID_HEADER: &str = "request-id";

#[derive(Debug, Clone)]
pub struct GoogleGenAiClient {
    http: reqwest::Client,
    api_key: String,
    base_url: String,
    max_retries: u32,
    initial_backoff: Duration,
    max_backoff: Duration,
}

impl GoogleGenAiClient {
    #[must_use]
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            http: build_http_client_or_default(),
            api_key: api_key.into(),
            base_url: read_base_url(),
            max_retries: DEFAULT_MAX_RETRIES,
            initial_backoff: DEFAULT_INITIAL_BACKOFF,
            max_backoff: DEFAULT_MAX_BACKOFF,
        }
    }

    pub fn from_env() -> Result<Self, ApiError> {
        let api_key = read_api_key()?
            .ok_or_else(|| ApiError::missing_credentials(PROVIDER_NAME, ENV_VARS))?;
        Ok(Self::new(api_key))
    }

    #[must_use]
    pub fn base_url(&self) -> &str {
        &self.base_url
    }

    #[must_use]
    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = base_url.into();
        self
    }

    pub async fn send_message(
        &self,
        request: &MessageRequest,
    ) -> Result<MessageResponse, ApiError> {
        let request = MessageRequest {
            stream: false,
            ..request.clone()
        };
        preflight_message_request(&request)?;
        let (response, effective_request) = self
            .post_with_max_tokens_adaptation(&request, false)
            .await?;
        let request_id = request_id_from_headers(response.headers());
        let body = response.text().await.map_err(ApiError::from)?;
        if let Some(api_error) = inline_error_response(&body, request_id.clone()) {
            return Err(api_error);
        }
        let payload = serde_json::from_str::<GenerateContentResponse>(&body).map_err(|error| {
            ApiError::json_deserialize(PROVIDER_NAME, &effective_request.model, &body, error)
        })?;
        let mut normalized = normalize_response(&effective_request.model, payload)?;
        if normalized.request_id.is_none() {
            normalized.request_id = request_id;
        }
        Ok(normalized)
    }

    async fn post_with_max_tokens_adaptation(
        &self,
        request: &MessageRequest,
        stream: bool,
    ) -> Result<(reqwest::Response, MessageRequest), ApiError> {
        match self.post_with_retry(request, stream).await {
            Ok(response) => Ok((response, request.clone())),
            Err(error) if error.is_max_tokens_exceeded() => {
                let downgraded =
                    super::downgrade_max_tokens_for_retry(&request.model, request.max_tokens);
                if downgraded >= request.max_tokens {
                    return Err(error);
                }
                let adjusted = MessageRequest {
                    max_tokens: downgraded,
                    ..request.clone()
                };
                let response = self.post_with_retry(&adjusted, stream).await?;
                Ok((response, adjusted))
            }
            Err(error) => Err(error),
        }
    }

    pub async fn stream_message(
        &self,
        request: &MessageRequest,
    ) -> Result<MessageStream, ApiError> {
        preflight_message_request(request)?;
        let streaming = MessageRequest {
            stream: true,
            ..request.clone()
        };
        let (response, _) = self
            .post_with_max_tokens_adaptation(&streaming, true)
            .await?;
        Ok(MessageStream {
            request_id: request_id_from_headers(response.headers()),
            response,
            parser: GoogleSseParser::default(),
            pending: VecDeque::new(),
            done: false,
            state: StreamState::new(streaming.model),
        })
    }

    async fn post_with_retry(
        &self,
        request: &MessageRequest,
        stream: bool,
    ) -> Result<reqwest::Response, ApiError> {
        let mut attempts = 0;
        let last_error = loop {
            attempts += 1;
            let retryable = match self.send_raw_request(request, stream).await {
                Ok(response) => match expect_success(response).await {
                    Ok(response) => return Ok(response),
                    Err(error) if error.is_retryable() && attempts <= self.max_retries + 1 => error,
                    Err(error) => return Err(error),
                },
                Err(error) if error.is_retryable() && attempts <= self.max_retries + 1 => error,
                Err(error) => return Err(error),
            };
            if attempts > self.max_retries {
                break retryable;
            }
            tokio::time::sleep(self.backoff_for_attempt(attempts)?).await;
        };
        Err(ApiError::RetriesExhausted {
            attempts,
            last_error: Box::new(last_error),
        })
    }

    async fn send_raw_request(
        &self,
        request: &MessageRequest,
        stream: bool,
    ) -> Result<reqwest::Response, ApiError> {
        let url = build_endpoint(&self.base_url, &request.model, stream);
        self.http
            .post(&url)
            .header("content-type", "application/json")
            .header("x-goog-api-key", &self.api_key)
            .json(&build_generate_content_request(request))
            .send()
            .await
            .map_err(ApiError::from)
    }

    fn backoff_for_attempt(&self, attempt: u32) -> Result<Duration, ApiError> {
        let Some(multiplier) = 1_u32.checked_shl(attempt.saturating_sub(1)) else {
            return Err(ApiError::BackoffOverflow {
                attempt,
                base_delay: self.initial_backoff,
            });
        };
        Ok(self
            .initial_backoff
            .checked_mul(multiplier)
            .map_or(self.max_backoff, |delay| delay.min(self.max_backoff)))
    }
}

impl Provider for GoogleGenAiClient {
    type Stream = MessageStream;

    fn send_message<'a>(
        &'a self,
        request: &'a MessageRequest,
    ) -> ProviderFuture<'a, MessageResponse> {
        Box::pin(async move { self.send_message(request).await })
    }

    fn stream_message<'a>(
        &'a self,
        request: &'a MessageRequest,
    ) -> ProviderFuture<'a, Self::Stream> {
        Box::pin(async move { self.stream_message(request).await })
    }
}

#[derive(Debug)]
pub struct MessageStream {
    request_id: Option<String>,
    response: reqwest::Response,
    parser: GoogleSseParser,
    pending: VecDeque<StreamEvent>,
    done: bool,
    state: StreamState,
}

impl MessageStream {
    #[must_use]
    pub fn request_id(&self) -> Option<&str> {
        self.request_id.as_deref()
    }

    pub async fn next_event(&mut self) -> Result<Option<StreamEvent>, ApiError> {
        loop {
            if let Some(event) = self.pending.pop_front() {
                return Ok(Some(event));
            }
            if self.done {
                self.pending.extend(self.state.finish());
                if let Some(event) = self.pending.pop_front() {
                    return Ok(Some(event));
                }
                return Ok(None);
            }
            match self.response.chunk().await? {
                Some(chunk) => {
                    for parsed in self.parser.push(&chunk)? {
                        self.pending.extend(self.state.ingest(parsed));
                    }
                }
                None => {
                    self.done = true;
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// SSE parsing
// ---------------------------------------------------------------------------

#[derive(Debug, Default)]
struct GoogleSseParser {
    buffer: Vec<u8>,
}

impl GoogleSseParser {
    fn push(&mut self, chunk: &[u8]) -> Result<Vec<GenerateContentResponse>, ApiError> {
        self.buffer.extend_from_slice(chunk);
        let mut events = Vec::new();
        while let Some(frame) = next_sse_frame(&mut self.buffer) {
            if let Some(payload) = parse_sse_frame(&frame)? {
                events.push(payload);
            }
        }
        Ok(events)
    }
}

fn next_sse_frame(buffer: &mut Vec<u8>) -> Option<String> {
    let separator = buffer
        .windows(2)
        .position(|window| window == b"\n\n")
        .map(|position| (position, 2))
        .or_else(|| {
            buffer
                .windows(4)
                .position(|window| window == b"\r\n\r\n")
                .map(|position| (position, 4))
        })?;
    let (position, separator_len) = separator;
    let frame = buffer.drain(..position + separator_len).collect::<Vec<_>>();
    let frame_len = frame.len().saturating_sub(separator_len);
    Some(String::from_utf8_lossy(&frame[..frame_len]).into_owned())
}

fn parse_sse_frame(frame: &str) -> Result<Option<GenerateContentResponse>, ApiError> {
    let trimmed = frame.trim();
    if trimmed.is_empty() {
        return Ok(None);
    }
    let mut data_lines = Vec::new();
    for line in trimmed.lines() {
        if line.starts_with(':') {
            continue;
        }
        if let Some(data) = line.strip_prefix("data:") {
            data_lines.push(data.trim_start());
        }
    }
    if data_lines.is_empty() {
        return Ok(None);
    }
    let payload = data_lines.join("\n");
    if payload == "[DONE]" {
        return Ok(None);
    }
    if let Some(api_error) = inline_error_response(&payload, None) {
        return Err(api_error);
    }
    serde_json::from_str::<GenerateContentResponse>(&payload)
        .map(Some)
        .map_err(|error| ApiError::json_deserialize(PROVIDER_NAME, "stream", &payload, error))
}

// ---------------------------------------------------------------------------
// Streaming state machine — translates Gemini chunks into Anthropic-shaped
// StreamEvents that the rest of the runtime understands.
// ---------------------------------------------------------------------------

#[derive(Debug)]
struct StreamState {
    model: String,
    message_id: String,
    message_started: bool,
    text_started: bool,
    text_finished: bool,
    finished: bool,
    next_block_index: u32,
    stop_reason: Option<String>,
    usage: Usage,
    /// Tracks whether we've forwarded any user-visible content (text or
    /// function call) so `finish` can decide whether to inject a fallback.
    /// Gemini 2.5 Flash sometimes returns `parts: []` with `finishReason:
    /// STOP`, which would otherwise crash the runtime with "produced no
    /// content".
    emitted_content: bool,
}

impl StreamState {
    fn new(model: String) -> Self {
        Self {
            model,
            message_id: format!("gemini_{}", short_id()),
            message_started: false,
            text_started: false,
            text_finished: false,
            finished: false,
            next_block_index: 0,
            stop_reason: None,
            usage: Usage::default(),
            emitted_content: false,
        }
    }

    fn ingest(&mut self, chunk: GenerateContentResponse) -> Vec<StreamEvent> {
        let mut events = Vec::new();
        if !self.message_started {
            self.message_started = true;
            events.push(StreamEvent::MessageStart(MessageStartEvent {
                message: MessageResponse {
                    id: self.message_id.clone(),
                    kind: "message".to_string(),
                    role: "assistant".to_string(),
                    content: Vec::new(),
                    model: self.model.clone(),
                    stop_reason: None,
                    stop_sequence: None,
                    usage: Usage::default(),
                    request_id: None,
                },
            }));
            // Open the text block immediately so downstream consumers can
            // attach text deltas without waiting for the first non-empty part.
            self.text_started = true;
            events.push(StreamEvent::ContentBlockStart(ContentBlockStartEvent {
                index: 0,
                content_block: OutputContentBlock::Text {
                    text: String::new(),
                },
            }));
            self.next_block_index = 1;
        }

        if let Some(usage) = chunk.usage_metadata {
            self.usage = Usage {
                input_tokens: usage.prompt_token_count.unwrap_or(0),
                cache_creation_input_tokens: 0,
                cache_read_input_tokens: usage.cached_content_token_count.unwrap_or(0),
                output_tokens: usage.candidates_token_count.unwrap_or(0),
            };
        }

        for candidate in chunk.candidates {
            if let Some(content) = candidate.content {
                for part in content.parts {
                    if let Some(text) = part.text.filter(|value| !value.is_empty()) {
                        self.emitted_content = true;
                        events.push(StreamEvent::ContentBlockDelta(ContentBlockDeltaEvent {
                            index: 0,
                            delta: ContentBlockDelta::TextDelta { text },
                        }));
                    }
                    if let Some(call) = part.function_call {
                        self.emitted_content = true;
                        let block_index = self.next_block_index;
                        self.next_block_index += 1;
                        let id = format!("call_{}_{}", call.name, block_index);
                        events.push(StreamEvent::ContentBlockStart(ContentBlockStartEvent {
                            index: block_index,
                            content_block: OutputContentBlock::ToolUse {
                                id,
                                name: call.name,
                                input: json!({}),
                            },
                        }));
                        let args_json = serde_json::to_string(&call.args.unwrap_or(Value::Null))
                            .unwrap_or_else(|_| "{}".to_string());
                        events.push(StreamEvent::ContentBlockDelta(ContentBlockDeltaEvent {
                            index: block_index,
                            delta: ContentBlockDelta::InputJsonDelta {
                                partial_json: args_json,
                            },
                        }));
                        events.push(StreamEvent::ContentBlockStop(ContentBlockStopEvent {
                            index: block_index,
                        }));
                    }
                }
            }
            if let Some(reason) = candidate.finish_reason {
                self.stop_reason = Some(normalize_finish_reason(&reason));
            }
        }
        events
    }

    fn finish(&mut self) -> Vec<StreamEvent> {
        if self.finished {
            return Vec::new();
        }
        self.finished = true;
        let mut events = Vec::new();
        // Gemini 2.5 sometimes ends a turn with `parts: []` and `finishReason:
        // STOP` — a literal empty assistant message. The runtime would then
        // refuse the turn with "produced no content". Inject a placeholder
        // text delta so the REPL stays alive and the user can simply ask
        // again with a clearer prompt.
        if self.message_started && !self.emitted_content {
            self.emitted_content = true;
            events.push(StreamEvent::ContentBlockDelta(ContentBlockDeltaEvent {
                index: 0,
                delta: ContentBlockDelta::TextDelta {
                    text: "(gemini returned an empty response — try rephrasing or asking again)"
                        .to_string(),
                },
            }));
        }
        if self.text_started && !self.text_finished {
            self.text_finished = true;
            events.push(StreamEvent::ContentBlockStop(ContentBlockStopEvent {
                index: 0,
            }));
        }
        if self.message_started {
            events.push(StreamEvent::MessageDelta(MessageDeltaEvent {
                delta: MessageDelta {
                    stop_reason: Some(
                        self.stop_reason
                            .clone()
                            .unwrap_or_else(|| "end_turn".to_string()),
                    ),
                    stop_sequence: None,
                },
                usage: self.usage.clone(),
            }));
            events.push(StreamEvent::MessageStop(MessageStopEvent {}));
        }
        events
    }
}

// ---------------------------------------------------------------------------
// Request translation
// ---------------------------------------------------------------------------

fn build_generate_content_request(request: &MessageRequest) -> Value {
    let mut payload = Map::new();

    if let Some(system) = request.system.as_ref().filter(|value| !value.is_empty()) {
        payload.insert(
            "systemInstruction".to_string(),
            json!({
                "parts": [{ "text": system }],
            }),
        );
    }

    payload.insert(
        "contents".to_string(),
        translate_messages(&request.messages),
    );

    let mut generation_config = Map::new();
    if request.max_tokens > 0 {
        generation_config.insert("maxOutputTokens".to_string(), json!(request.max_tokens));
    }
    if let Some(temperature) = request.temperature {
        generation_config.insert("temperature".to_string(), json!(temperature));
    }
    if let Some(top_p) = request.top_p {
        generation_config.insert("topP".to_string(), json!(top_p));
    }
    if let Some(stop) = &request.stop {
        if !stop.is_empty() {
            generation_config.insert("stopSequences".to_string(), json!(stop));
        }
    }
    if !generation_config.is_empty() {
        payload.insert(
            "generationConfig".to_string(),
            Value::Object(generation_config),
        );
    }

    if let Some(tools) = &request.tools {
        if !tools.is_empty() {
            payload.insert(
                "tools".to_string(),
                json!([{
                    "functionDeclarations": tools.iter().map(translate_tool_definition).collect::<Vec<_>>(),
                }]),
            );
        }
    }
    if let Some(choice) = &request.tool_choice {
        payload.insert("toolConfig".to_string(), translate_tool_choice(choice));
    }

    Value::Object(payload)
}

fn translate_messages(messages: &[InputMessage]) -> Value {
    // Build a name lookup so tool_results can be paired back to the function
    // name Gemini originally emitted. Anthropic addresses tool results by
    // tool_use_id; Gemini addresses them by function name.
    let mut id_to_name: std::collections::HashMap<String, String> =
        std::collections::HashMap::new();
    for message in messages {
        for block in &message.content {
            if let InputContentBlock::ToolUse { id, name, .. } = block {
                id_to_name.insert(id.clone(), name.clone());
            }
        }
    }

    let mut contents = Vec::new();
    for message in messages {
        let role = match message.role.as_str() {
            "assistant" => "model",
            _ => "user",
        };
        let mut parts = Vec::new();
        for block in &message.content {
            match block {
                InputContentBlock::Text { text } if !text.is_empty() => {
                    parts.push(json!({ "text": text }));
                }
                InputContentBlock::Text { .. } => {}
                InputContentBlock::ToolUse { name, input, .. } => {
                    parts.push(json!({
                        "functionCall": {
                            "name": name,
                            "args": input,
                        }
                    }));
                }
                InputContentBlock::ToolResult {
                    tool_use_id,
                    content,
                    is_error,
                } => {
                    let name = id_to_name
                        .get(tool_use_id)
                        .cloned()
                        .unwrap_or_else(|| tool_use_id.clone());
                    let response_value = tool_result_payload(content, *is_error);
                    parts.push(json!({
                        "functionResponse": {
                            "name": name,
                            "response": response_value,
                        }
                    }));
                }
            }
        }
        if parts.is_empty() {
            continue;
        }
        contents.push(json!({
            "role": role,
            "parts": parts,
        }));
    }
    Value::Array(contents)
}

fn tool_result_payload(blocks: &[ToolResultContentBlock], is_error: bool) -> Value {
    let mut joined_text = String::new();
    let mut json_payload: Option<Value> = None;
    for block in blocks {
        match block {
            ToolResultContentBlock::Text { text } => {
                if !joined_text.is_empty() {
                    joined_text.push('\n');
                }
                joined_text.push_str(text);
            }
            ToolResultContentBlock::Json { value } => {
                json_payload = Some(value.clone());
            }
        }
    }
    let mut response = Map::new();
    if let Some(value) = json_payload {
        response.insert("result".to_string(), value);
    }
    if !joined_text.is_empty() {
        response.insert("content".to_string(), Value::String(joined_text));
    }
    if response.is_empty() {
        response.insert("content".to_string(), Value::String(String::new()));
    }
    if is_error {
        response.insert("error".to_string(), Value::Bool(true));
    }
    Value::Object(response)
}

fn translate_tool_definition(tool: &ToolDefinition) -> Value {
    let mut entry = Map::new();
    entry.insert("name".to_string(), Value::String(tool.name.clone()));
    if let Some(desc) = &tool.description {
        entry.insert("description".to_string(), Value::String(desc.clone()));
    }
    let mut parameters = tool.input_schema.clone();
    sanitize_schema_for_gemini(&mut parameters);
    entry.insert("parameters".to_string(), parameters);
    Value::Object(entry)
}

/// Coerce a JSON-Schema fragment into the OpenAPI 3.0 subset Gemini accepts.
///
/// Gemini's Schema proto is much narrower than the JSON-Schema dialect that
/// Anthropic and OpenAI both consume:
///   - `type` must be a single string, never an array (`["string","null"]`).
///     Nullability is expressed via a separate `nullable: true` field.
///   - `additionalProperties`, `$schema`, `definitions`, `$defs`, `$ref`,
///     `const`, `examples`, `default`, `title`, `pattern`, and a handful of
///     other fields are silently rejected with `Unknown name "<field>"`.
///   - `oneOf` / `anyOf` / `allOf` are not part of the proto. We collapse to
///     the first variant so the call can still go through; this loses some
///     fidelity but is far better than erroring out the whole request.
///   - `format` only accepts `enum`, `date-time`, etc. for specific types;
///     we strip it on non-string types to be safe.
///
/// This is intentionally aggressive — claw's built-in tool schemas come
/// straight from JSON-Schema and would otherwise blow up the entire request
/// when even one tool has a nullable property.
fn sanitize_schema_for_gemini(schema: &mut Value) {
    // Collapse anyOf/oneOf/allOf to their first variant before walking the
    // node so we don't leave the unsupported fields behind on the parent.
    for combinator in ["anyOf", "oneOf", "allOf"] {
        let collapsed = schema
            .as_object_mut()
            .and_then(|obj| obj.remove(combinator))
            .and_then(|value| match value {
                Value::Array(mut arr) if !arr.is_empty() => Some(arr.remove(0)),
                _ => None,
            });
        if let Some(replacement) = collapsed {
            // Merge the collapsed variant onto the current node, letting
            // existing keys win so we don't clobber anything explicit.
            if let (Some(target), Value::Object(source)) = (schema.as_object_mut(), replacement) {
                for (key, value) in source {
                    target.entry(key).or_insert(value);
                }
            }
        }
    }

    if let Some(obj) = schema.as_object_mut() {
        // Strip fields Gemini's proto parser does not recognise.
        for key in [
            "$schema",
            "$id",
            "$ref",
            "$defs",
            "definitions",
            "additionalProperties",
            "unevaluatedProperties",
            "patternProperties",
            "examples",
            "default",
            "title",
            "const",
            "exclusiveMinimum",
            "exclusiveMaximum",
            "multipleOf",
            "contentEncoding",
            "contentMediaType",
        ] {
            obj.remove(key);
        }

        // Coerce `type: ["string","null"]` into `type: "string"` + nullable.
        if let Some(type_value) = obj.get("type").cloned() {
            if let Value::Array(types) = type_value {
                let mut nullable = false;
                let mut concrete: Option<String> = None;
                for entry in types {
                    if let Some(s) = entry.as_str() {
                        if s == "null" {
                            nullable = true;
                        } else if concrete.is_none() {
                            concrete = Some(s.to_string());
                        }
                    }
                }
                if let Some(t) = concrete {
                    obj.insert("type".to_string(), Value::String(t));
                } else {
                    obj.remove("type");
                }
                if nullable {
                    obj.insert("nullable".to_string(), Value::Bool(true));
                }
            }
        }

        // `format` is only valid on string/integer/number per Gemini's schema;
        // even there, only a small allow-list works (`enum`, `date-time`,
        // `int32`, `int64`, `float`, `double`). Strip aggressively rather
        // than maintain a duplicate allow-list of our own.
        if obj.get("type").and_then(Value::as_str) != Some("string")
            && obj.get("type").and_then(Value::as_str) != Some("integer")
            && obj.get("type").and_then(Value::as_str) != Some("number")
        {
            obj.remove("format");
        }

        if let Some(props) = obj.get_mut("properties").and_then(Value::as_object_mut) {
            for value in props.values_mut() {
                sanitize_schema_for_gemini(value);
            }
        }
        if let Some(items) = obj.get_mut("items") {
            sanitize_schema_for_gemini(items);
        }
        // Some tool schemas wrap the parameters in an `enum` array of objects;
        // recurse defensively into common nested holders.
        if let Some(prefix_items) = obj.get_mut("prefixItems") {
            sanitize_schema_for_gemini(prefix_items);
        }
    }
}

fn translate_tool_choice(choice: &ToolChoice) -> Value {
    let mode = match choice {
        ToolChoice::Auto => "AUTO",
        ToolChoice::Any => "ANY",
        ToolChoice::Tool { .. } => "ANY",
    };
    let mut config = json!({
        "functionCallingConfig": { "mode": mode },
    });
    if let ToolChoice::Tool { name } = choice {
        config["functionCallingConfig"]["allowedFunctionNames"] = json!([name]);
    }
    config
}

// ---------------------------------------------------------------------------
// Response translation
// ---------------------------------------------------------------------------

fn normalize_response(
    model: &str,
    response: GenerateContentResponse,
) -> Result<MessageResponse, ApiError> {
    let candidate = response
        .candidates
        .into_iter()
        .next()
        .ok_or(ApiError::InvalidSseFrame(
            "google generate-content response missing candidates",
        ))?;

    let mut content_blocks = Vec::new();
    let mut next_id = 0_u32;
    if let Some(content) = candidate.content {
        for part in content.parts {
            if let Some(text) = part.text.filter(|value| !value.is_empty()) {
                content_blocks.push(OutputContentBlock::Text { text });
            }
            if let Some(call) = part.function_call {
                next_id += 1;
                let id = format!("call_{}_{}", call.name, next_id);
                content_blocks.push(OutputContentBlock::ToolUse {
                    id,
                    name: call.name,
                    input: call.args.unwrap_or(Value::Null),
                });
            }
        }
    }

    let usage = response
        .usage_metadata
        .map(|usage| Usage {
            input_tokens: usage.prompt_token_count.unwrap_or(0),
            cache_creation_input_tokens: 0,
            cache_read_input_tokens: usage.cached_content_token_count.unwrap_or(0),
            output_tokens: usage.candidates_token_count.unwrap_or(0),
        })
        .unwrap_or_default();

    Ok(MessageResponse {
        id: format!("gemini_{}", short_id()),
        kind: "message".to_string(),
        role: "assistant".to_string(),
        content: content_blocks,
        model: model.to_string(),
        stop_reason: candidate
            .finish_reason
            .map(|reason| normalize_finish_reason(&reason)),
        stop_sequence: None,
        usage,
        request_id: None,
    })
}

#[derive(Debug, Deserialize)]
struct GenerateContentResponse {
    #[serde(default)]
    candidates: Vec<Candidate>,
    #[serde(default, rename = "usageMetadata")]
    usage_metadata: Option<UsageMetadata>,
}

#[derive(Debug, Deserialize)]
struct Candidate {
    #[serde(default)]
    content: Option<CandidateContent>,
    #[serde(default, rename = "finishReason")]
    finish_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
struct CandidateContent {
    #[serde(default)]
    parts: Vec<CandidatePart>,
}

#[derive(Debug, Deserialize)]
struct CandidatePart {
    #[serde(default)]
    text: Option<String>,
    #[serde(default, rename = "functionCall")]
    function_call: Option<FunctionCall>,
}

#[derive(Debug, Deserialize)]
struct FunctionCall {
    name: String,
    #[serde(default)]
    args: Option<Value>,
}

#[derive(Debug, Deserialize)]
struct UsageMetadata {
    #[serde(default, rename = "promptTokenCount")]
    prompt_token_count: Option<u32>,
    #[serde(default, rename = "candidatesTokenCount")]
    candidates_token_count: Option<u32>,
    #[serde(default, rename = "cachedContentTokenCount")]
    cached_content_token_count: Option<u32>,
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn build_endpoint(base_url: &str, model: &str, stream: bool) -> String {
    let trimmed_base = base_url.trim_end_matches('/');
    let wire_model = strip_routing_prefix(model);
    let action = if stream {
        "streamGenerateContent?alt=sse"
    } else {
        "generateContent"
    };
    format!("{trimmed_base}/models/{wire_model}:{action}")
}

/// Strip routing prefixes (`google/`, `gemini/`) so the wire URL gets the
/// bare model id. `gemini-2.5-flash` is left untouched.
fn strip_routing_prefix(model: &str) -> &str {
    if let Some(rest) = model.strip_prefix("google/") {
        return rest;
    }
    if let Some(rest) = model.strip_prefix("gemini/") {
        return rest;
    }
    model
}

#[must_use]
pub fn read_base_url() -> String {
    std::env::var("GEMINI_BASE_URL")
        .ok()
        .or_else(|| std::env::var("GOOGLE_BASE_URL").ok())
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| DEFAULT_BASE_URL.to_string())
}

#[must_use]
pub fn has_credentials() -> bool {
    read_api_key().ok().flatten().is_some()
}

fn read_api_key() -> Result<Option<String>, ApiError> {
    for key in ENV_VARS {
        if let Some(value) = read_env_non_empty(key)? {
            return Ok(Some(value));
        }
    }
    Ok(None)
}

fn read_env_non_empty(key: &str) -> Result<Option<String>, ApiError> {
    match std::env::var(key) {
        Ok(value) if !value.is_empty() => Ok(Some(value)),
        Ok(_) | Err(std::env::VarError::NotPresent) => Ok(super::dotenv_value(key)),
        Err(error) => Err(ApiError::from(error)),
    }
}

fn request_id_from_headers(headers: &reqwest::header::HeaderMap) -> Option<String> {
    headers
        .get(REQUEST_ID_HEADER)
        .or_else(|| headers.get(ALT_REQUEST_ID_HEADER))
        .and_then(|value| value.to_str().ok())
        .map(ToOwned::to_owned)
}

async fn expect_success(response: reqwest::Response) -> Result<reqwest::Response, ApiError> {
    let status = response.status();
    if status.is_success() {
        return Ok(response);
    }
    let request_id = request_id_from_headers(response.headers());
    let body = response.text().await.unwrap_or_default();
    let parsed = serde_json::from_str::<Value>(&body).ok();
    let (error_type, message) = parsed
        .as_ref()
        .and_then(|value| value.get("error"))
        .map(|err| {
            (
                err.get("status").and_then(Value::as_str).map(str::to_owned),
                err.get("message")
                    .and_then(Value::as_str)
                    .map(str::to_owned),
            )
        })
        .unwrap_or((None, None));
    Err(ApiError::Api {
        status,
        error_type,
        message,
        request_id,
        body,
        retryable: is_retryable_status(status),
    })
}

fn inline_error_response(body: &str, request_id: Option<String>) -> Option<ApiError> {
    let value = serde_json::from_str::<Value>(body).ok()?;
    let err = value.get("error")?;
    let message = err
        .get("message")
        .and_then(Value::as_str)
        .map(str::to_owned);
    let status_code = err.get("code").and_then(Value::as_u64).map(|c| c as u16);
    let status = status_code
        .and_then(|code| reqwest::StatusCode::from_u16(code).ok())
        .unwrap_or(reqwest::StatusCode::BAD_REQUEST);
    Some(ApiError::Api {
        status,
        error_type: err.get("status").and_then(Value::as_str).map(str::to_owned),
        message,
        request_id,
        body: body.to_string(),
        retryable: false,
    })
}

const fn is_retryable_status(status: reqwest::StatusCode) -> bool {
    matches!(status.as_u16(), 408 | 409 | 429 | 500 | 502 | 503 | 504)
}

fn normalize_finish_reason(value: &str) -> String {
    match value.to_ascii_uppercase().as_str() {
        "STOP" | "FINISH_REASON_STOP" => "end_turn".to_string(),
        "MAX_TOKENS" => "max_tokens".to_string(),
        "SAFETY" | "RECITATION" => "stop_sequence".to_string(),
        other => other.to_ascii_lowercase(),
    }
}

fn short_id() -> String {
    use std::sync::atomic::{AtomicU64, Ordering};
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0);
    let tick = COUNTER.fetch_add(1, Ordering::Relaxed);
    format!("{:x}{:x}", now, tick)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn build_endpoint_uses_generate_content_for_unary_calls() {
        let url = build_endpoint(
            "https://generativelanguage.googleapis.com/v1beta/",
            "gemini-2.5-flash",
            false,
        );
        assert_eq!(
            url,
            "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent"
        );
    }

    #[test]
    fn build_endpoint_uses_stream_path_for_streaming_calls() {
        let url = build_endpoint(
            "https://core-ai-platform.ikameglobal.com/gemini/v1beta",
            "gemini-2.5-pro",
            true,
        );
        assert_eq!(
            url,
            "https://core-ai-platform.ikameglobal.com/gemini/v1beta/models/gemini-2.5-pro:streamGenerateContent?alt=sse"
        );
    }

    #[test]
    fn strip_routing_prefix_removes_namespace() {
        assert_eq!(
            strip_routing_prefix("google/gemini-2.5-flash"),
            "gemini-2.5-flash"
        );
        assert_eq!(
            strip_routing_prefix("gemini/gemini-2.5-pro"),
            "gemini-2.5-pro"
        );
        assert_eq!(strip_routing_prefix("gemini-2.5-flash"), "gemini-2.5-flash");
    }

    #[test]
    fn translate_messages_maps_user_assistant_and_tool_results() {
        let messages = vec![
            InputMessage {
                role: "user".to_string(),
                content: vec![InputContentBlock::Text {
                    text: "hello".to_string(),
                }],
            },
            InputMessage {
                role: "assistant".to_string(),
                content: vec![InputContentBlock::ToolUse {
                    id: "call_1".to_string(),
                    name: "lookup".to_string(),
                    input: json!({ "q": "weather" }),
                }],
            },
            InputMessage {
                role: "user".to_string(),
                content: vec![InputContentBlock::ToolResult {
                    tool_use_id: "call_1".to_string(),
                    content: vec![ToolResultContentBlock::Text {
                        text: "sunny".to_string(),
                    }],
                    is_error: false,
                }],
            },
        ];
        let translated = translate_messages(&messages);
        let array = translated.as_array().expect("array");
        assert_eq!(array.len(), 3);
        assert_eq!(array[0]["role"], "user");
        assert_eq!(array[0]["parts"][0]["text"], "hello");
        assert_eq!(array[1]["role"], "model");
        assert_eq!(array[1]["parts"][0]["functionCall"]["name"], "lookup");
        assert_eq!(array[1]["parts"][0]["functionCall"]["args"]["q"], "weather");
        assert_eq!(array[2]["role"], "user");
        assert_eq!(array[2]["parts"][0]["functionResponse"]["name"], "lookup");
        assert_eq!(
            array[2]["parts"][0]["functionResponse"]["response"]["content"],
            "sunny"
        );
    }

    #[test]
    fn build_request_includes_system_instruction_and_generation_config() {
        let request = MessageRequest {
            model: "gemini-2.5-flash".to_string(),
            max_tokens: 256,
            messages: vec![InputMessage::user_text("hi")],
            system: Some("be terse".to_string()),
            temperature: Some(0.5),
            ..Default::default()
        };
        let payload = build_generate_content_request(&request);
        assert_eq!(payload["systemInstruction"]["parts"][0]["text"], "be terse");
        assert_eq!(payload["generationConfig"]["maxOutputTokens"], 256);
        assert_eq!(payload["generationConfig"]["temperature"], 0.5);
        assert_eq!(payload["contents"][0]["role"], "user");
        assert_eq!(payload["contents"][0]["parts"][0]["text"], "hi");
    }

    #[test]
    fn sanitize_schema_strips_unsupported_fields() {
        let mut schema = json!({
            "$schema": "http://json-schema.org/draft-07/schema#",
            "type": "object",
            "additionalProperties": false,
            "properties": {
                "city": { "type": "string", "additionalProperties": false }
            }
        });
        sanitize_schema_for_gemini(&mut schema);
        assert!(schema.get("$schema").is_none());
        assert!(schema.get("additionalProperties").is_none());
        assert!(schema["properties"]["city"]
            .get("additionalProperties")
            .is_none());
        assert_eq!(schema["properties"]["city"]["type"], "string");
    }

    #[test]
    fn sanitize_schema_collapses_nullable_type_arrays() {
        // Gemini's proto parser rejects `type: ["string","null"]` outright;
        // we have to coerce to `type: "string"` + `nullable: true`.
        let mut schema = json!({
            "type": "object",
            "properties": {
                "name": { "type": ["string", "null"] },
                "age": { "type": ["null", "integer"] },
            }
        });
        sanitize_schema_for_gemini(&mut schema);
        assert_eq!(schema["properties"]["name"]["type"], "string");
        assert_eq!(schema["properties"]["name"]["nullable"], true);
        assert_eq!(schema["properties"]["age"]["type"], "integer");
        assert_eq!(schema["properties"]["age"]["nullable"], true);
    }

    #[test]
    fn sanitize_schema_collapses_combinators_to_first_variant() {
        let mut schema = json!({
            "anyOf": [
                { "type": "string" },
                { "type": "null" }
            ]
        });
        sanitize_schema_for_gemini(&mut schema);
        assert!(schema.get("anyOf").is_none());
        assert_eq!(schema["type"], "string");
    }

    #[test]
    fn normalize_response_extracts_text_and_usage() {
        let response = GenerateContentResponse {
            candidates: vec![Candidate {
                content: Some(CandidateContent {
                    parts: vec![CandidatePart {
                        text: Some("hello there".to_string()),
                        function_call: None,
                    }],
                }),
                finish_reason: Some("STOP".to_string()),
            }],
            usage_metadata: Some(UsageMetadata {
                prompt_token_count: Some(12),
                candidates_token_count: Some(3),
                cached_content_token_count: Some(0),
            }),
        };
        let normalized = normalize_response("gemini-2.5-flash", response).expect("normalize");
        assert_eq!(normalized.content.len(), 1);
        match &normalized.content[0] {
            OutputContentBlock::Text { text } => assert_eq!(text, "hello there"),
            other => panic!("expected text block, got {other:?}"),
        }
        assert_eq!(normalized.stop_reason.as_deref(), Some("end_turn"));
        assert_eq!(normalized.usage.input_tokens, 12);
        assert_eq!(normalized.usage.output_tokens, 3);
    }

    #[test]
    fn parse_sse_frame_extracts_payload() {
        let frame = "data: {\"candidates\": [{\"content\": {\"parts\": [{\"text\": \"hi\"}]}}]}\n";
        let parsed = parse_sse_frame(frame).expect("parse").expect("payload");
        assert_eq!(parsed.candidates.len(), 1);
        assert_eq!(
            parsed.candidates[0].content.as_ref().unwrap().parts[0]
                .text
                .as_deref(),
            Some("hi")
        );
    }

    #[test]
    fn stream_state_emits_fallback_when_gemini_returns_empty_stop() {
        // Regression: gemini-2.5-flash sometimes returns
        //   { content: { role: "model" }, finishReason: "STOP" }
        // with `parts: []`. The runtime would otherwise crash the turn with
        // "assistant stream produced no content". Confirm the stream state
        // injects a placeholder text delta so the REPL stays alive.
        let mut state = StreamState::new("gemini-2.5-flash".to_string());
        let chunk = GenerateContentResponse {
            candidates: vec![Candidate {
                content: Some(CandidateContent { parts: vec![] }),
                finish_reason: Some("STOP".to_string()),
            }],
            usage_metadata: Some(UsageMetadata {
                prompt_token_count: Some(4155),
                candidates_token_count: None,
                cached_content_token_count: None,
            }),
        };
        let mut events = state.ingest(chunk);
        events.extend(state.finish());

        let has_text_delta = events.iter().any(|event| {
            matches!(
                event,
                StreamEvent::ContentBlockDelta(ContentBlockDeltaEvent {
                    delta: ContentBlockDelta::TextDelta { text },
                    ..
                }) if text.contains("empty response")
            )
        });
        assert!(
            has_text_delta,
            "expected fallback text delta to be emitted, got events: {events:?}"
        );

        let has_stop = events
            .iter()
            .any(|event| matches!(event, StreamEvent::MessageStop(_)));
        assert!(has_stop, "expected MessageStop event, got: {events:?}");
    }
}
