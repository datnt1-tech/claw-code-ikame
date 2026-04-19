#![allow(clippy::cast_possible_truncation)]
use std::future::Future;
use std::pin::Pin;

use serde::Serialize;

use crate::error::ApiError;
use crate::types::{MessageRequest, MessageResponse};

pub mod anthropic;
pub mod google_genai;
pub mod openai_compat;

#[allow(dead_code)]
pub type ProviderFuture<'a, T> = Pin<Box<dyn Future<Output = Result<T, ApiError>> + Send + 'a>>;

#[allow(dead_code)]
pub trait Provider {
    type Stream;

    fn send_message<'a>(
        &'a self,
        request: &'a MessageRequest,
    ) -> ProviderFuture<'a, MessageResponse>;

    fn stream_message<'a>(
        &'a self,
        request: &'a MessageRequest,
    ) -> ProviderFuture<'a, Self::Stream>;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProviderKind {
    Anthropic,
    Xai,
    OpenAi,
    GoogleGenAi,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ProviderMetadata {
    pub provider: ProviderKind,
    pub auth_env: &'static str,
    pub base_url_env: &'static str,
    pub default_base_url: &'static str,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ModelTokenLimit {
    pub max_output_tokens: u32,
    pub context_window_tokens: u32,
}

const MODEL_REGISTRY: &[(&str, ProviderMetadata)] = &[
    (
        "opus",
        ProviderMetadata {
            provider: ProviderKind::Anthropic,
            auth_env: "ANTHROPIC_API_KEY",
            base_url_env: "ANTHROPIC_BASE_URL",
            default_base_url: anthropic::DEFAULT_BASE_URL,
        },
    ),
    (
        "sonnet",
        ProviderMetadata {
            provider: ProviderKind::Anthropic,
            auth_env: "ANTHROPIC_API_KEY",
            base_url_env: "ANTHROPIC_BASE_URL",
            default_base_url: anthropic::DEFAULT_BASE_URL,
        },
    ),
    (
        "haiku",
        ProviderMetadata {
            provider: ProviderKind::Anthropic,
            auth_env: "ANTHROPIC_API_KEY",
            base_url_env: "ANTHROPIC_BASE_URL",
            default_base_url: anthropic::DEFAULT_BASE_URL,
        },
    ),
    (
        "grok",
        ProviderMetadata {
            provider: ProviderKind::Xai,
            auth_env: "XAI_API_KEY",
            base_url_env: "XAI_BASE_URL",
            default_base_url: openai_compat::DEFAULT_XAI_BASE_URL,
        },
    ),
    (
        "grok-3",
        ProviderMetadata {
            provider: ProviderKind::Xai,
            auth_env: "XAI_API_KEY",
            base_url_env: "XAI_BASE_URL",
            default_base_url: openai_compat::DEFAULT_XAI_BASE_URL,
        },
    ),
    (
        "grok-mini",
        ProviderMetadata {
            provider: ProviderKind::Xai,
            auth_env: "XAI_API_KEY",
            base_url_env: "XAI_BASE_URL",
            default_base_url: openai_compat::DEFAULT_XAI_BASE_URL,
        },
    ),
    (
        "grok-3-mini",
        ProviderMetadata {
            provider: ProviderKind::Xai,
            auth_env: "XAI_API_KEY",
            base_url_env: "XAI_BASE_URL",
            default_base_url: openai_compat::DEFAULT_XAI_BASE_URL,
        },
    ),
    (
        "grok-2",
        ProviderMetadata {
            provider: ProviderKind::Xai,
            auth_env: "XAI_API_KEY",
            base_url_env: "XAI_BASE_URL",
            default_base_url: openai_compat::DEFAULT_XAI_BASE_URL,
        },
    ),
    (
        "deepseek",
        ProviderMetadata {
            provider: ProviderKind::OpenAi,
            auth_env: "DEEPSEEK_API_KEY",
            base_url_env: "DEEPSEEK_BASE_URL",
            default_base_url: openai_compat::DEFAULT_DEEPSEEK_BASE_URL,
        },
    ),
    (
        "r1",
        ProviderMetadata {
            provider: ProviderKind::OpenAi,
            auth_env: "DEEPSEEK_API_KEY",
            base_url_env: "DEEPSEEK_BASE_URL",
            default_base_url: openai_compat::DEFAULT_DEEPSEEK_BASE_URL,
        },
    ),
];

#[must_use]
pub fn resolve_model_alias(model: &str) -> String {
    let trimmed = model.trim();
    let lower = trimmed.to_ascii_lowercase();
    MODEL_REGISTRY
        .iter()
        .find_map(|(alias, metadata)| {
            (*alias == lower).then_some(match metadata.provider {
                ProviderKind::Anthropic => match *alias {
                    "opus" => "claude-opus-4-6",
                    "sonnet" => "claude-sonnet-4-6",
                    "haiku" => "claude-haiku-4-5-20251213",
                    _ => trimmed,
                },
                ProviderKind::Xai => match *alias {
                    "grok" | "grok-3" => "grok-3",
                    "grok-mini" | "grok-3-mini" => "grok-3-mini",
                    "grok-2" => "grok-2",
                    _ => trimmed,
                },
                ProviderKind::OpenAi => match *alias {
                    "deepseek" => "deepseek-chat",
                    "r1" => "deepseek-reasoner",
                    _ => trimmed,
                },
                ProviderKind::GoogleGenAi => trimmed,
            })
        })
        .map_or_else(|| trimmed.to_string(), ToOwned::to_owned)
}

#[must_use]
pub fn metadata_for_model(model: &str) -> Option<ProviderMetadata> {
    let canonical = resolve_model_alias(model);
    if canonical.starts_with("claude") {
        return Some(ProviderMetadata {
            provider: ProviderKind::Anthropic,
            auth_env: "ANTHROPIC_API_KEY",
            base_url_env: "ANTHROPIC_BASE_URL",
            default_base_url: anthropic::DEFAULT_BASE_URL,
        });
    }
    if canonical.starts_with("grok") {
        return Some(ProviderMetadata {
            provider: ProviderKind::Xai,
            auth_env: "XAI_API_KEY",
            base_url_env: "XAI_BASE_URL",
            default_base_url: openai_compat::DEFAULT_XAI_BASE_URL,
        });
    }
    // Explicit provider-namespaced models (e.g. "openai/gpt-4.1-mini") must
    // route to the correct provider regardless of which auth env vars are set.
    // Without this, detect_provider_kind falls through to the auth-sniffer
    // order and misroutes to Anthropic if ANTHROPIC_API_KEY is present.
    if canonical.starts_with("openai/") || canonical.starts_with("gpt-") {
        return Some(ProviderMetadata {
            provider: ProviderKind::OpenAi,
            auth_env: "OPENAI_API_KEY",
            base_url_env: "OPENAI_BASE_URL",
            default_base_url: openai_compat::DEFAULT_OPENAI_BASE_URL,
        });
    }
    // Alibaba DashScope compatible-mode endpoint. Routes qwen/* and bare
    // qwen-* model names (qwen-max, qwen-plus, qwen-turbo, qwen-qwq, etc.)
    // to the OpenAI-compat client pointed at DashScope's /compatible-mode/v1.
    // Uses the OpenAi provider kind because DashScope speaks the OpenAI REST
    // shape — only the base URL and auth env var differ.
    if canonical.starts_with("qwen/") || canonical.starts_with("qwen-") {
        return Some(ProviderMetadata {
            provider: ProviderKind::OpenAi,
            auth_env: "DASHSCOPE_API_KEY",
            base_url_env: "DASHSCOPE_BASE_URL",
            default_base_url: openai_compat::DEFAULT_DASHSCOPE_BASE_URL,
        });
    }
    // DeepSeek official API. Routes deepseek/* and bare deepseek-* model
    // names (deepseek-chat, deepseek-reasoner) to the OpenAI-compat client
    // pointed at api.deepseek.com/v1. Uses the OpenAi provider kind because
    // DeepSeek speaks the OpenAI REST shape.
    if canonical.starts_with("deepseek/") || canonical.starts_with("deepseek-") {
        return Some(ProviderMetadata {
            provider: ProviderKind::OpenAi,
            auth_env: "DEEPSEEK_API_KEY",
            base_url_env: "DEEPSEEK_BASE_URL",
            default_base_url: openai_compat::DEFAULT_DEEPSEEK_BASE_URL,
        });
    }
    // Google Generative AI / Gemini family. Routes any model name with a
    // gemini-/google/-style prefix to the native Gemini REST backend so users
    // can point GEMINI_BASE_URL at the official endpoint or any compatible
    // proxy (LiteLLM, internal company gateways, etc.).
    if canonical.starts_with("gemini-")
        || canonical.starts_with("gemini/")
        || canonical.starts_with("google/")
    {
        return Some(ProviderMetadata {
            provider: ProviderKind::GoogleGenAi,
            auth_env: "GEMINI_API_KEY",
            base_url_env: "GEMINI_BASE_URL",
            default_base_url: google_genai::DEFAULT_BASE_URL,
        });
    }
    None
}

#[must_use]
pub fn detect_provider_kind(model: &str) -> ProviderKind {
    if let Some(metadata) = metadata_for_model(model) {
        return metadata.provider;
    }
    // When OPENAI_BASE_URL is set, the user explicitly configured an
    // OpenAI-compatible endpoint. Prefer it over the Anthropic fallback
    // even when the model name has no recognized prefix — this is the
    // common case for local providers (Ollama, LM Studio, vLLM, etc.)
    // where model names like "qwen2.5-coder:7b" don't match any prefix.
    if std::env::var_os("OPENAI_BASE_URL").is_some() && openai_compat::has_api_key("OPENAI_API_KEY")
    {
        return ProviderKind::OpenAi;
    }
    if anthropic::has_auth_from_env_or_saved().unwrap_or(false) {
        return ProviderKind::Anthropic;
    }
    if openai_compat::has_api_key("OPENAI_API_KEY") {
        return ProviderKind::OpenAi;
    }
    if openai_compat::has_api_key("XAI_API_KEY") {
        return ProviderKind::Xai;
    }
    if google_genai::has_credentials() {
        return ProviderKind::GoogleGenAi;
    }
    // Last resort: if OPENAI_BASE_URL is set without OPENAI_API_KEY (some
    // local providers like Ollama don't require auth), still route there.
    if std::env::var_os("OPENAI_BASE_URL").is_some() {
        return ProviderKind::OpenAi;
    }
    ProviderKind::Anthropic
}

#[must_use]
pub fn max_tokens_for_model(model: &str) -> u32 {
    let base = model_token_limit(model).map_or_else(
        || {
            let canonical = resolve_model_alias(model);
            if canonical.contains("opus") {
                32_000
            } else {
                64_000
            }
        },
        |limit| limit.max_output_tokens,
    );
    // Apply the global env override even for models that didn't match the
    // registry (model_token_limit already applies it internally, but the
    // heuristic fallback above bypasses that path).
    env_max_output_tokens_override().map_or(base, |cap| base.min(cap))
}

/// Returns the effective max output tokens for a model, preferring a plugin
/// override when present. Falls back to [`max_tokens_for_model`] when the
/// override is `None`.
#[must_use]
pub fn max_tokens_for_model_with_override(model: &str, plugin_override: Option<u32>) -> u32 {
    plugin_override.unwrap_or_else(|| max_tokens_for_model(model))
}

#[must_use]
pub fn model_token_limit(model: &str) -> Option<ModelTokenLimit> {
    let canonical = resolve_model_alias(model);
    let limit = exact_model_token_limit(canonical.as_str())
        .or_else(|| prefix_model_token_limit(canonical.as_str()));
    limit.map(|mut lim| {
        if let Some(cap) = env_max_output_tokens_override() {
            lim.max_output_tokens = lim.max_output_tokens.min(cap);
        }
        lim
    })
}

/// Exact-match limits for the specific model revisions we've verified.
/// Keep conservative: if a provider later raises a cap, add a new entry
/// rather than guessing here. The prefix fallback catches anything unknown.
fn exact_model_token_limit(canonical: &str) -> Option<ModelTokenLimit> {
    match canonical {
        "claude-opus-4-6" => Some(ModelTokenLimit {
            max_output_tokens: 32_000,
            context_window_tokens: 200_000,
        }),
        "claude-sonnet-4-6" | "claude-haiku-4-5-20251213" => Some(ModelTokenLimit {
            max_output_tokens: 64_000,
            context_window_tokens: 200_000,
        }),
        "grok-3" | "grok-3-mini" => Some(ModelTokenLimit {
            max_output_tokens: 64_000,
            context_window_tokens: 131_072,
        }),
        "grok-2" => Some(ModelTokenLimit {
            max_output_tokens: 32_000,
            context_window_tokens: 131_072,
        }),
        // DeepSeek caps max_tokens at 8192 and provides a 128k context window
        // for both deepseek-chat (V3) and deepseek-reasoner (R1). Requests
        // above 8192 output tokens are rejected with 400 invalid_request_error.
        "deepseek-chat" | "deepseek-reasoner" => Some(ModelTokenLimit {
            max_output_tokens: 8_192,
            context_window_tokens: 131_072,
        }),
        "gpt-4o"
        | "gpt-4o-2024-11-20"
        | "gpt-4o-2024-08-06"
        | "gpt-4o-2024-05-13"
        | "gpt-4o-mini"
        | "gpt-4o-mini-2024-07-18" => Some(ModelTokenLimit {
            max_output_tokens: 16_384,
            context_window_tokens: 128_000,
        }),
        "gpt-4.1" | "gpt-4.1-mini" | "gpt-4.1-nano" => Some(ModelTokenLimit {
            max_output_tokens: 32_768,
            context_window_tokens: 1_047_576,
        }),
        "gpt-4-turbo" | "gpt-4-turbo-preview" | "gpt-4-turbo-2024-04-09" => Some(ModelTokenLimit {
            max_output_tokens: 4_096,
            context_window_tokens: 128_000,
        }),
        "gpt-3.5-turbo" | "gpt-3.5-turbo-0125" => Some(ModelTokenLimit {
            max_output_tokens: 4_096,
            context_window_tokens: 16_385,
        }),
        "o1" | "o1-preview" | "o1-2024-12-17" => Some(ModelTokenLimit {
            max_output_tokens: 100_000,
            context_window_tokens: 200_000,
        }),
        "o1-mini" | "o3-mini" | "o4-mini" => Some(ModelTokenLimit {
            max_output_tokens: 65_536,
            context_window_tokens: 200_000,
        }),
        "o3" => Some(ModelTokenLimit {
            max_output_tokens: 100_000,
            context_window_tokens: 200_000,
        }),
        "gemini-1.5-pro" | "gemini-1.5-pro-latest" | "gemini-1.5-pro-002" => {
            Some(ModelTokenLimit {
                max_output_tokens: 8_192,
                context_window_tokens: 2_000_000,
            })
        }
        "gemini-1.5-flash"
        | "gemini-1.5-flash-latest"
        | "gemini-1.5-flash-002"
        | "gemini-1.5-flash-8b" => Some(ModelTokenLimit {
            max_output_tokens: 8_192,
            context_window_tokens: 1_048_576,
        }),
        "gemini-2.0-flash" | "gemini-2.0-flash-001" | "gemini-2.0-flash-exp" => {
            Some(ModelTokenLimit {
                max_output_tokens: 8_192,
                context_window_tokens: 1_048_576,
            })
        }
        "gemini-2.5-pro" | "gemini-2.5-pro-preview" | "gemini-2.5-pro-exp" => {
            Some(ModelTokenLimit {
                max_output_tokens: 65_536,
                context_window_tokens: 2_097_152,
            })
        }
        "gemini-2.5-flash" | "gemini-2.5-flash-preview" | "gemini-2.5-flash-exp" => {
            Some(ModelTokenLimit {
                max_output_tokens: 65_536,
                context_window_tokens: 1_048_576,
            })
        }
        "qwen-max" | "qwen-max-latest" => Some(ModelTokenLimit {
            max_output_tokens: 8_192,
            context_window_tokens: 32_768,
        }),
        "qwen-plus" | "qwen-plus-latest" => Some(ModelTokenLimit {
            max_output_tokens: 8_192,
            context_window_tokens: 131_072,
        }),
        "qwen-turbo" | "qwen-turbo-latest" => Some(ModelTokenLimit {
            max_output_tokens: 8_192,
            context_window_tokens: 1_000_000,
        }),
        "qwen3-coder" | "qwen3-coder-plus" => Some(ModelTokenLimit {
            max_output_tokens: 8_192,
            context_window_tokens: 131_072,
        }),
        _ => None,
    }
}

/// Prefix-based fallback so every known model family has sensible caps
/// without enumerating every revision. Conservative defaults — prefer to
/// under-estimate max_output than over-estimate (easier to surface the
/// low-cap warning and trigger auto-downgrade on provider 400 errors).
fn prefix_model_token_limit(canonical: &str) -> Option<ModelTokenLimit> {
    if canonical.starts_with("claude-opus") {
        return Some(ModelTokenLimit {
            max_output_tokens: 32_000,
            context_window_tokens: 200_000,
        });
    }
    if canonical.starts_with("claude-") {
        return Some(ModelTokenLimit {
            max_output_tokens: 64_000,
            context_window_tokens: 200_000,
        });
    }
    if canonical.starts_with("grok-") {
        return Some(ModelTokenLimit {
            max_output_tokens: 64_000,
            context_window_tokens: 131_072,
        });
    }
    if canonical.starts_with("deepseek-") || canonical.starts_with("deepseek/") {
        return Some(ModelTokenLimit {
            max_output_tokens: 8_192,
            context_window_tokens: 131_072,
        });
    }
    if canonical.starts_with("o1-") || canonical.starts_with("o3-") || canonical.starts_with("o4-")
    {
        return Some(ModelTokenLimit {
            max_output_tokens: 65_536,
            context_window_tokens: 200_000,
        });
    }
    if canonical.starts_with("gpt-4o") {
        return Some(ModelTokenLimit {
            max_output_tokens: 16_384,
            context_window_tokens: 128_000,
        });
    }
    if canonical.starts_with("gpt-4.1") {
        return Some(ModelTokenLimit {
            max_output_tokens: 32_768,
            context_window_tokens: 1_047_576,
        });
    }
    if canonical.starts_with("gpt-4") {
        return Some(ModelTokenLimit {
            max_output_tokens: 4_096,
            context_window_tokens: 128_000,
        });
    }
    if canonical.starts_with("gpt-3.5") {
        return Some(ModelTokenLimit {
            max_output_tokens: 4_096,
            context_window_tokens: 16_385,
        });
    }
    if canonical.starts_with("openai/") {
        return Some(ModelTokenLimit {
            max_output_tokens: 16_384,
            context_window_tokens: 128_000,
        });
    }
    if canonical.starts_with("gemini-2.5") {
        return Some(ModelTokenLimit {
            max_output_tokens: 65_536,
            context_window_tokens: 1_048_576,
        });
    }
    if canonical.starts_with("gemini-")
        || canonical.starts_with("gemini/")
        || canonical.starts_with("google/")
    {
        return Some(ModelTokenLimit {
            max_output_tokens: 8_192,
            context_window_tokens: 1_048_576,
        });
    }
    if canonical.starts_with("qwen-")
        || canonical.starts_with("qwen/")
        || canonical.starts_with("qwen3")
    {
        return Some(ModelTokenLimit {
            max_output_tokens: 8_192,
            context_window_tokens: 131_072,
        });
    }
    None
}

/// Reads `CLAW_MAX_OUTPUT_TOKENS` and parses it as a positive u32. Provides
/// a global override so users can defensively cap output for any model
/// (e.g. to 4096 when running behind a proxy with its own stricter cap)
/// without needing a code change or provider-specific config.
fn env_max_output_tokens_override() -> Option<u32> {
    std::env::var("CLAW_MAX_OUTPUT_TOKENS")
        .ok()
        .and_then(|raw| raw.trim().parse::<u32>().ok())
        .filter(|value| *value > 0)
}

/// Compute a safe `max_tokens` value after a provider rejected the
/// request because the requested cap exceeds the model's limit. If the
/// registry knows the real cap, drop to it; otherwise halve the current
/// request with a 1024-token floor. Never returns 0.
#[must_use]
pub fn downgrade_max_tokens_for_retry(model: &str, current_max: u32) -> u32 {
    if let Some(limit) = model_token_limit(model) {
        if current_max > limit.max_output_tokens {
            return limit.max_output_tokens.max(1);
        }
    }
    (current_max / 2).max(1024)
}

pub fn preflight_message_request(request: &MessageRequest) -> Result<(), ApiError> {
    let Some(limit) = model_token_limit(&request.model) else {
        return Ok(());
    };

    let estimated_input_tokens = estimate_message_request_input_tokens(request);
    let estimated_total_tokens = estimated_input_tokens.saturating_add(request.max_tokens);
    if estimated_total_tokens > limit.context_window_tokens {
        return Err(ApiError::ContextWindowExceeded {
            model: resolve_model_alias(&request.model),
            estimated_input_tokens,
            requested_output_tokens: request.max_tokens,
            estimated_total_tokens,
            context_window_tokens: limit.context_window_tokens,
        });
    }

    Ok(())
}

fn estimate_message_request_input_tokens(request: &MessageRequest) -> u32 {
    let mut estimate = estimate_serialized_tokens(&request.messages);
    estimate = estimate.saturating_add(estimate_serialized_tokens(&request.system));
    estimate = estimate.saturating_add(estimate_serialized_tokens(&request.tools));
    estimate = estimate.saturating_add(estimate_serialized_tokens(&request.tool_choice));
    estimate
}

fn estimate_serialized_tokens<T: Serialize>(value: &T) -> u32 {
    serde_json::to_vec(value)
        .ok()
        .map_or(0, |bytes| (bytes.len() / 4 + 1) as u32)
}

/// Env var names used by other provider backends. When Anthropic auth
/// resolution fails we sniff these so we can hint the user that their
/// credentials probably belong to a different provider and suggest the
/// model-prefix routing fix that would select it.
const FOREIGN_PROVIDER_ENV_VARS: &[(&str, &str, &str)] = &[
    (
        "OPENAI_API_KEY",
        "OpenAI-compat",
        "prefix your model name with `openai/` (e.g. `--model openai/gpt-4.1-mini`) so prefix routing selects the OpenAI-compatible provider, and set `OPENAI_BASE_URL` if you are pointing at OpenRouter/Ollama/a local server",
    ),
    (
        "XAI_API_KEY",
        "xAI",
        "use an xAI model alias (e.g. `--model grok` or `--model grok-mini`) so the prefix router selects the xAI backend",
    ),
    (
        "DASHSCOPE_API_KEY",
        "Alibaba DashScope",
        "prefix your model name with `qwen/` or `qwen-` (e.g. `--model qwen-plus`) so prefix routing selects the DashScope backend",
    ),
    (
        "DEEPSEEK_API_KEY",
        "DeepSeek",
        "use a DeepSeek model alias (e.g. `--model deepseek-chat` or `--model deepseek-reasoner`) so the prefix router selects the DeepSeek backend, and set `DEEPSEEK_BASE_URL` if you are pointing at a custom proxy",
    ),
    (
        "GEMINI_API_KEY",
        "Google Generative AI (Gemini)",
        "use a `gemini-` model alias (e.g. `--model gemini-2.5-flash`) so prefix routing selects the Gemini backend, and set `GEMINI_BASE_URL` if you are pointing at a custom proxy",
    ),
    (
        "GOOGLE_API_KEY",
        "Google Generative AI (Gemini)",
        "use a `gemini-` model alias (e.g. `--model gemini-2.5-flash`) so prefix routing selects the Gemini backend, and set `GEMINI_BASE_URL` if you are pointing at a custom proxy",
    ),
];

/// Check whether an env var is set to a non-empty value either in the real
/// process environment or in the working-directory `.env` file. Mirrors the
/// credential discovery path used by `read_env_non_empty` so the hint text
/// stays truthful when users rely on `.env` instead of a real export.
fn env_or_dotenv_present(key: &str) -> bool {
    match std::env::var(key) {
        Ok(value) if !value.is_empty() => true,
        Ok(_) | Err(std::env::VarError::NotPresent) => {
            dotenv_value(key).is_some_and(|value| !value.is_empty())
        }
        Err(_) => false,
    }
}

/// Produce a hint string describing the first foreign provider credential
/// that is present in the environment when Anthropic auth resolution has
/// just failed. Returns `None` when no foreign credential is set, in which
/// case the caller should fall back to the plain `missing_credentials`
/// error without a hint.
pub(crate) fn anthropic_missing_credentials_hint() -> Option<String> {
    for (env_var, provider_label, fix_hint) in FOREIGN_PROVIDER_ENV_VARS {
        if env_or_dotenv_present(env_var) {
            return Some(format!(
                "I see {env_var} is set — if you meant to use the {provider_label} provider, {fix_hint}."
            ));
        }
    }
    None
}

/// Build an Anthropic-specific `MissingCredentials` error, attaching a
/// hint suggesting the probable fix whenever a different provider's
/// credentials are already present in the environment. Anthropic call
/// sites should prefer this helper over `ApiError::missing_credentials`
/// so users who mistyped a model name or forgot the prefix get a useful
/// signal instead of a generic "missing Anthropic credentials" wall.
pub(crate) fn anthropic_missing_credentials() -> ApiError {
    const PROVIDER: &str = "Anthropic";
    const ENV_VARS: &[&str] = &["ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_API_KEY"];
    match anthropic_missing_credentials_hint() {
        Some(hint) => ApiError::missing_credentials_with_hint(PROVIDER, ENV_VARS, hint),
        None => ApiError::missing_credentials(PROVIDER, ENV_VARS),
    }
}

/// Parse a `.env` file body into key/value pairs using a minimal `KEY=VALUE`
/// grammar. Lines that are blank, start with `#`, or do not contain `=` are
/// ignored. Surrounding double or single quotes are stripped from the value.
/// An optional leading `export ` prefix on the key is also stripped so files
/// shared with shell `source` workflows still parse cleanly.
pub(crate) fn parse_dotenv(content: &str) -> std::collections::HashMap<String, String> {
    let mut values = std::collections::HashMap::new();
    for raw_line in content.lines() {
        let line = raw_line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let Some((raw_key, raw_value)) = line.split_once('=') else {
            continue;
        };
        let trimmed_key = raw_key.trim();
        let key = trimmed_key
            .strip_prefix("export ")
            .map_or(trimmed_key, str::trim)
            .to_string();
        if key.is_empty() {
            continue;
        }
        let trimmed_value = raw_value.trim();
        let unquoted = if (trimmed_value.starts_with('"') && trimmed_value.ends_with('"')
            || trimmed_value.starts_with('\'') && trimmed_value.ends_with('\''))
            && trimmed_value.len() >= 2
        {
            &trimmed_value[1..trimmed_value.len() - 1]
        } else {
            trimmed_value
        };
        values.insert(key, unquoted.to_string());
    }
    values
}

/// Load and parse a `.env` file from the given path. Missing files yield
/// `None` instead of an error so callers can use this as a soft fallback.
pub(crate) fn load_dotenv_file(
    path: &std::path::Path,
) -> Option<std::collections::HashMap<String, String>> {
    let content = std::fs::read_to_string(path).ok()?;
    Some(parse_dotenv(&content))
}

/// Look up `key` in a `.env` file located in the current working directory.
/// Returns `None` when the file is missing, the key is absent, or the value
/// is empty.
pub(crate) fn dotenv_value(key: &str) -> Option<String> {
    let cwd = std::env::current_dir().ok()?;
    let values = load_dotenv_file(&cwd.join(".env"))?;
    values.get(key).filter(|value| !value.is_empty()).cloned()
}

#[cfg(test)]
mod tests {
    use std::ffi::OsString;
    use std::sync::{Mutex, OnceLock};

    use serde_json::json;

    use crate::error::ApiError;
    use crate::types::{
        InputContentBlock, InputMessage, MessageRequest, ToolChoice, ToolDefinition,
    };

    use super::{
        anthropic_missing_credentials, anthropic_missing_credentials_hint, detect_provider_kind,
        downgrade_max_tokens_for_retry, load_dotenv_file, max_tokens_for_model,
        max_tokens_for_model_with_override, model_token_limit, parse_dotenv,
        preflight_message_request, resolve_model_alias, ProviderKind,
    };

    /// Serializes every test in this module that mutates process-wide
    /// environment variables so concurrent test threads cannot observe
    /// each other's partially-applied state while probing the foreign
    /// provider credential sniffer.
    fn env_lock() -> std::sync::MutexGuard<'static, ()> {
        static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        LOCK.get_or_init(|| Mutex::new(()))
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    /// Snapshot-restore guard for a single environment variable. Captures
    /// the original value on construction, applies the requested override
    /// (set or remove), and restores the original on drop so tests leave
    /// the process env untouched even when they panic mid-assertion.
    struct EnvVarGuard {
        key: &'static str,
        original: Option<OsString>,
    }

    impl EnvVarGuard {
        fn set(key: &'static str, value: Option<&str>) -> Self {
            let original = std::env::var_os(key);
            match value {
                Some(value) => std::env::set_var(key, value),
                None => std::env::remove_var(key),
            }
            Self { key, original }
        }
    }

    impl Drop for EnvVarGuard {
        fn drop(&mut self) {
            match self.original.take() {
                Some(value) => std::env::set_var(self.key, value),
                None => std::env::remove_var(self.key),
            }
        }
    }

    #[test]
    fn resolves_grok_aliases() {
        assert_eq!(resolve_model_alias("grok"), "grok-3");
        assert_eq!(resolve_model_alias("grok-mini"), "grok-3-mini");
        assert_eq!(resolve_model_alias("grok-2"), "grok-2");
    }

    #[test]
    fn detects_provider_from_model_name_first() {
        assert_eq!(detect_provider_kind("grok"), ProviderKind::Xai);
        assert_eq!(
            detect_provider_kind("claude-sonnet-4-6"),
            ProviderKind::Anthropic
        );
    }

    #[test]
    fn openai_namespaced_model_routes_to_openai_not_anthropic() {
        // Regression: "openai/gpt-4.1-mini" was misrouted to Anthropic when
        // ANTHROPIC_API_KEY was set because metadata_for_model returned None
        // and detect_provider_kind fell through to auth-sniffer order.
        // The model prefix must win over env-var presence.
        let kind = super::metadata_for_model("openai/gpt-4.1-mini").map_or_else(
            || detect_provider_kind("openai/gpt-4.1-mini"),
            |m| m.provider,
        );
        assert_eq!(
            kind,
            ProviderKind::OpenAi,
            "openai/ prefix must route to OpenAi regardless of ANTHROPIC_API_KEY"
        );

        // Also cover bare gpt- prefix
        let kind2 = super::metadata_for_model("gpt-4o")
            .map_or_else(|| detect_provider_kind("gpt-4o"), |m| m.provider);
        assert_eq!(kind2, ProviderKind::OpenAi);
    }

    #[test]
    fn qwen_prefix_routes_to_dashscope_not_anthropic() {
        // User request from Discord #clawcode-get-help: web3g wants to use
        // Qwen 3.6 Plus via native Alibaba DashScope API (not OpenRouter,
        // which has lower rate limits). metadata_for_model must route
        // qwen/* and bare qwen-* to the OpenAi provider kind pointed at
        // the DashScope compatible-mode endpoint, regardless of whether
        // ANTHROPIC_API_KEY is present in the environment.
        let meta = super::metadata_for_model("qwen/qwen-max")
            .expect("qwen/ prefix must resolve to DashScope metadata");
        assert_eq!(meta.provider, ProviderKind::OpenAi);
        assert_eq!(meta.auth_env, "DASHSCOPE_API_KEY");
        assert_eq!(meta.base_url_env, "DASHSCOPE_BASE_URL");
        assert!(meta.default_base_url.contains("dashscope.aliyuncs.com"));

        // Bare qwen- prefix also routes
        let meta2 = super::metadata_for_model("qwen-plus")
            .expect("qwen- prefix must resolve to DashScope metadata");
        assert_eq!(meta2.provider, ProviderKind::OpenAi);
        assert_eq!(meta2.auth_env, "DASHSCOPE_API_KEY");

        // detect_provider_kind must agree even if ANTHROPIC_API_KEY is set
        let kind = detect_provider_kind("qwen/qwen3-coder");
        assert_eq!(
            kind,
            ProviderKind::OpenAi,
            "qwen/ prefix must win over auth-sniffer order"
        );
    }

    #[test]
    fn keeps_existing_max_token_heuristic() {
        assert_eq!(max_tokens_for_model("opus"), 32_000);
        assert_eq!(max_tokens_for_model("grok-3"), 64_000);
    }

    #[test]
    fn plugin_config_max_output_tokens_overrides_model_default() {
        // given
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("time should be after epoch")
            .as_nanos();
        let root = std::env::temp_dir().join(format!("api-plugin-max-tokens-{nanos}"));
        let cwd = root.join("project");
        let home = root.join("home").join(".claw");
        std::fs::create_dir_all(cwd.join(".claw")).expect("project config dir");
        std::fs::create_dir_all(&home).expect("home config dir");
        std::fs::write(
            home.join("settings.json"),
            r#"{
              "plugins": {
                "maxOutputTokens": 12345
              }
            }"#,
        )
        .expect("write plugin settings");

        // when
        let loaded = runtime::ConfigLoader::new(&cwd, &home)
            .load()
            .expect("config should load");
        let plugin_override = loaded.plugins().max_output_tokens();
        let effective = max_tokens_for_model_with_override("claude-opus-4-6", plugin_override);

        // then
        assert_eq!(plugin_override, Some(12345));
        assert_eq!(effective, 12345);
        assert_ne!(effective, max_tokens_for_model("claude-opus-4-6"));

        std::fs::remove_dir_all(root).expect("cleanup temp dir");
    }

    #[test]
    fn max_tokens_for_model_with_override_falls_back_when_plugin_unset() {
        // given
        let plugin_override: Option<u32> = None;

        // when
        let effective = max_tokens_for_model_with_override("claude-opus-4-6", plugin_override);

        // then
        assert_eq!(effective, max_tokens_for_model("claude-opus-4-6"));
        assert_eq!(effective, 32_000);
    }

    #[test]
    fn model_token_limit_covers_new_model_families() {
        let _guard = env_lock();
        // Clear any stray override so the assertions see the raw caps.
        let _override = EnvVarGuard::set("CLAW_MAX_OUTPUT_TOKENS", None);

        let gpt4o = model_token_limit("gpt-4o").expect("gpt-4o registered");
        assert_eq!(gpt4o.max_output_tokens, 16_384);
        assert_eq!(gpt4o.context_window_tokens, 128_000);

        let turbo = model_token_limit("gpt-4-turbo").expect("gpt-4-turbo registered");
        assert_eq!(turbo.max_output_tokens, 4_096);

        let gpt41 = model_token_limit("gpt-4.1").expect("gpt-4.1 registered");
        assert_eq!(gpt41.max_output_tokens, 32_768);

        let flash = model_token_limit("gemini-1.5-flash").expect("gemini-1.5-flash registered");
        assert_eq!(flash.max_output_tokens, 8_192);
        assert!(flash.context_window_tokens >= 1_000_000);

        let gem25 = model_token_limit("gemini-2.5-pro").expect("gemini-2.5-pro registered");
        assert_eq!(gem25.max_output_tokens, 65_536);

        let qwen = model_token_limit("qwen-max").expect("qwen-max registered");
        assert_eq!(qwen.max_output_tokens, 8_192);
    }

    #[test]
    fn model_token_limit_falls_back_on_family_prefix() {
        let _guard = env_lock();
        let _override = EnvVarGuard::set("CLAW_MAX_OUTPUT_TOKENS", None);

        // Unknown revision should still route through the prefix fallback.
        let future_sonnet = model_token_limit("claude-sonnet-5-1")
            .expect("claude- prefix should cover future revs");
        assert_eq!(future_sonnet.max_output_tokens, 64_000);

        let future_opus = model_token_limit("claude-opus-5")
            .expect("claude-opus prefix should cover future revs");
        assert_eq!(future_opus.max_output_tokens, 32_000);

        let openai_ns = model_token_limit("openai/gpt-custom")
            .expect("openai/ prefix should have a fallback cap");
        assert_eq!(openai_ns.max_output_tokens, 16_384);

        let gemini_exp = model_token_limit("gemini-3.0-flash-thinking")
            .expect("gemini- prefix should have a fallback cap");
        assert_eq!(gemini_exp.max_output_tokens, 8_192);

        let qwen_ns = model_token_limit("qwen/qwen-max-longcontext")
            .expect("qwen/ prefix should have a fallback cap");
        assert_eq!(qwen_ns.max_output_tokens, 8_192);
    }

    #[test]
    fn env_override_caps_max_tokens_for_any_model() {
        let _guard = env_lock();
        let _override = EnvVarGuard::set("CLAW_MAX_OUTPUT_TOKENS", Some("2048"));

        // Registered model → env override clamps down.
        assert_eq!(max_tokens_for_model("claude-opus-4-6"), 2_048);
        assert_eq!(
            model_token_limit("claude-opus-4-6")
                .expect("registered")
                .max_output_tokens,
            2_048
        );

        // DeepSeek's native cap (8192) is already below the override value,
        // but we still compute min(cap, override) so 8192 stays the ceiling.
        let _relax = EnvVarGuard::set("CLAW_MAX_OUTPUT_TOKENS", Some("100000"));
        assert_eq!(max_tokens_for_model("deepseek-chat"), 8_192);
    }

    #[test]
    fn downgrade_uses_registered_cap_when_request_over_limit() {
        // DeepSeek hard cap is 8192; a request asking for 64k should be
        // downgraded to 8192, not halved to 32k.
        assert_eq!(
            downgrade_max_tokens_for_retry("deepseek-chat", 64_000),
            8_192
        );
        assert_eq!(downgrade_max_tokens_for_retry("gpt-4-turbo", 16_384), 4_096);
    }

    #[test]
    fn downgrade_halves_when_request_already_under_registered_cap() {
        // Already at or below the registered cap — halve with 1024 floor.
        assert_eq!(
            downgrade_max_tokens_for_retry("claude-opus-4-6", 4_096),
            2_048
        );
        assert_eq!(
            downgrade_max_tokens_for_retry("claude-opus-4-6", 1_500),
            1_024
        );
    }

    #[test]
    fn returns_context_window_metadata_for_supported_models() {
        assert_eq!(
            model_token_limit("claude-sonnet-4-6")
                .expect("claude-sonnet-4-6 should be registered")
                .context_window_tokens,
            200_000
        );
        assert_eq!(
            model_token_limit("grok-mini")
                .expect("grok-mini should resolve to a registered model")
                .context_window_tokens,
            131_072
        );
    }

    #[test]
    fn preflight_blocks_requests_that_exceed_the_model_context_window() {
        let request = MessageRequest {
            model: "claude-sonnet-4-6".to_string(),
            max_tokens: 64_000,
            messages: vec![InputMessage {
                role: "user".to_string(),
                content: vec![InputContentBlock::Text {
                    text: "x".repeat(600_000),
                }],
            }],
            system: Some("Keep the answer short.".to_string()),
            tools: Some(vec![ToolDefinition {
                name: "weather".to_string(),
                description: Some("Fetches weather".to_string()),
                input_schema: json!({
                    "type": "object",
                    "properties": { "city": { "type": "string" } },
                }),
            }]),
            tool_choice: Some(ToolChoice::Auto),
            stream: true,
            ..Default::default()
        };

        let error = preflight_message_request(&request)
            .expect_err("oversized request should be rejected before the provider call");

        match error {
            ApiError::ContextWindowExceeded {
                model,
                estimated_input_tokens,
                requested_output_tokens,
                estimated_total_tokens,
                context_window_tokens,
            } => {
                assert_eq!(model, "claude-sonnet-4-6");
                assert!(estimated_input_tokens > 136_000);
                assert_eq!(requested_output_tokens, 64_000);
                assert!(estimated_total_tokens > context_window_tokens);
                assert_eq!(context_window_tokens, 200_000);
            }
            other => panic!("expected context-window preflight failure, got {other:?}"),
        }
    }

    #[test]
    fn preflight_skips_unknown_models() {
        let request = MessageRequest {
            model: "unknown-model".to_string(),
            max_tokens: 64_000,
            messages: vec![InputMessage {
                role: "user".to_string(),
                content: vec![InputContentBlock::Text {
                    text: "x".repeat(600_000),
                }],
            }],
            system: None,
            tools: None,
            tool_choice: None,
            stream: false,
            ..Default::default()
        };

        preflight_message_request(&request)
            .expect("models without context metadata should skip the guarded preflight");
    }

    #[test]
    fn parse_dotenv_extracts_keys_handles_comments_quotes_and_export_prefix() {
        // given
        let body = "\
# this is a comment

ANTHROPIC_API_KEY=plain-value
XAI_API_KEY=\"quoted-value\"
OPENAI_API_KEY='single-quoted'
export GROK_API_KEY=exported-value
   PADDED_KEY  =  padded-value  
EMPTY_VALUE=
NO_EQUALS_LINE
";

        // when
        let values = parse_dotenv(body);

        // then
        assert_eq!(
            values.get("ANTHROPIC_API_KEY").map(String::as_str),
            Some("plain-value")
        );
        assert_eq!(
            values.get("XAI_API_KEY").map(String::as_str),
            Some("quoted-value")
        );
        assert_eq!(
            values.get("OPENAI_API_KEY").map(String::as_str),
            Some("single-quoted")
        );
        assert_eq!(
            values.get("GROK_API_KEY").map(String::as_str),
            Some("exported-value")
        );
        assert_eq!(
            values.get("PADDED_KEY").map(String::as_str),
            Some("padded-value")
        );
        assert_eq!(values.get("EMPTY_VALUE").map(String::as_str), Some(""));
        assert!(!values.contains_key("NO_EQUALS_LINE"));
        assert!(!values.contains_key("# this is a comment"));
    }

    #[test]
    fn load_dotenv_file_reads_keys_from_disk_and_returns_none_when_missing() {
        // given
        let temp_root = std::env::temp_dir().join(format!(
            "api-dotenv-test-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map_or(0, |duration| duration.as_nanos())
        ));
        std::fs::create_dir_all(&temp_root).expect("create temp dir");
        let env_path = temp_root.join(".env");
        std::fs::write(
            &env_path,
            "ANTHROPIC_API_KEY=secret-from-file\n# comment\nXAI_API_KEY=\"xai-secret\"\n",
        )
        .expect("write .env");
        let missing_path = temp_root.join("does-not-exist.env");

        // when
        let loaded = load_dotenv_file(&env_path).expect("file should load");
        let missing = load_dotenv_file(&missing_path);

        // then
        assert_eq!(
            loaded.get("ANTHROPIC_API_KEY").map(String::as_str),
            Some("secret-from-file")
        );
        assert_eq!(
            loaded.get("XAI_API_KEY").map(String::as_str),
            Some("xai-secret")
        );
        assert!(missing.is_none());

        let _ = std::fs::remove_dir_all(&temp_root);
    }

    #[test]
    fn anthropic_missing_credentials_hint_is_none_when_no_foreign_creds_present() {
        // given
        let _lock = env_lock();
        let _openai = EnvVarGuard::set("OPENAI_API_KEY", None);
        let _xai = EnvVarGuard::set("XAI_API_KEY", None);
        let _dashscope = EnvVarGuard::set("DASHSCOPE_API_KEY", None);

        // when
        let hint = anthropic_missing_credentials_hint();

        // then
        assert!(
            hint.is_none(),
            "no hint should be produced when every foreign provider env var is absent, got {hint:?}"
        );
    }

    #[test]
    fn anthropic_missing_credentials_hint_detects_openai_api_key_and_recommends_openai_prefix() {
        // given
        let _lock = env_lock();
        let _openai = EnvVarGuard::set("OPENAI_API_KEY", Some("sk-openrouter-varleg"));
        let _xai = EnvVarGuard::set("XAI_API_KEY", None);
        let _dashscope = EnvVarGuard::set("DASHSCOPE_API_KEY", None);

        // when
        let hint = anthropic_missing_credentials_hint()
            .expect("OPENAI_API_KEY presence should produce a hint");

        // then
        assert!(
            hint.contains("OPENAI_API_KEY is set"),
            "hint should name the detected env var so users recognize it: {hint}"
        );
        assert!(
            hint.contains("OpenAI-compat"),
            "hint should identify the target provider: {hint}"
        );
        assert!(
            hint.contains("openai/"),
            "hint should mention the `openai/` prefix routing fix: {hint}"
        );
        assert!(
            hint.contains("OPENAI_BASE_URL"),
            "hint should mention OPENAI_BASE_URL so OpenRouter users see the full picture: {hint}"
        );
    }

    #[test]
    fn anthropic_missing_credentials_hint_detects_xai_api_key() {
        // given
        let _lock = env_lock();
        let _openai = EnvVarGuard::set("OPENAI_API_KEY", None);
        let _xai = EnvVarGuard::set("XAI_API_KEY", Some("xai-test-key"));
        let _dashscope = EnvVarGuard::set("DASHSCOPE_API_KEY", None);

        // when
        let hint = anthropic_missing_credentials_hint()
            .expect("XAI_API_KEY presence should produce a hint");

        // then
        assert!(
            hint.contains("XAI_API_KEY is set"),
            "hint should name XAI_API_KEY: {hint}"
        );
        assert!(
            hint.contains("xAI"),
            "hint should identify the xAI provider: {hint}"
        );
        assert!(
            hint.contains("grok"),
            "hint should suggest a grok-prefixed model alias: {hint}"
        );
    }

    #[test]
    fn anthropic_missing_credentials_hint_detects_dashscope_api_key() {
        // given
        let _lock = env_lock();
        let _openai = EnvVarGuard::set("OPENAI_API_KEY", None);
        let _xai = EnvVarGuard::set("XAI_API_KEY", None);
        let _dashscope = EnvVarGuard::set("DASHSCOPE_API_KEY", Some("sk-dashscope-test"));

        // when
        let hint = anthropic_missing_credentials_hint()
            .expect("DASHSCOPE_API_KEY presence should produce a hint");

        // then
        assert!(
            hint.contains("DASHSCOPE_API_KEY is set"),
            "hint should name DASHSCOPE_API_KEY: {hint}"
        );
        assert!(
            hint.contains("DashScope"),
            "hint should identify the DashScope provider: {hint}"
        );
        assert!(
            hint.contains("qwen"),
            "hint should suggest a qwen-prefixed model alias: {hint}"
        );
    }

    #[test]
    fn anthropic_missing_credentials_hint_prefers_openai_when_multiple_foreign_creds_set() {
        // given
        let _lock = env_lock();
        let _openai = EnvVarGuard::set("OPENAI_API_KEY", Some("sk-openrouter-varleg"));
        let _xai = EnvVarGuard::set("XAI_API_KEY", Some("xai-test-key"));
        let _dashscope = EnvVarGuard::set("DASHSCOPE_API_KEY", Some("sk-dashscope-test"));

        // when
        let hint = anthropic_missing_credentials_hint()
            .expect("multiple foreign creds should still produce a hint");

        // then
        assert!(
            hint.contains("OPENAI_API_KEY"),
            "OpenAI should be prioritized because it is the most common misrouting pattern (OpenRouter users), got: {hint}"
        );
        assert!(
            !hint.contains("XAI_API_KEY"),
            "only the first detected provider should be named to keep the hint focused, got: {hint}"
        );
    }

    #[test]
    fn anthropic_missing_credentials_builds_error_with_canonical_env_vars_and_no_hint_when_clean() {
        // given
        let _lock = env_lock();
        let _openai = EnvVarGuard::set("OPENAI_API_KEY", None);
        let _xai = EnvVarGuard::set("XAI_API_KEY", None);
        let _dashscope = EnvVarGuard::set("DASHSCOPE_API_KEY", None);

        // when
        let error = anthropic_missing_credentials();

        // then
        match &error {
            ApiError::MissingCredentials {
                provider,
                env_vars,
                hint,
            } => {
                assert_eq!(*provider, "Anthropic");
                assert_eq!(*env_vars, &["ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_API_KEY"]);
                assert!(
                    hint.is_none(),
                    "clean environment should not generate a hint, got {hint:?}"
                );
            }
            other => panic!("expected MissingCredentials variant, got {other:?}"),
        }
        let rendered = error.to_string();
        assert!(
            !rendered.contains(" — hint: "),
            "rendered error should be a plain missing-creds message: {rendered}"
        );
    }

    #[test]
    fn anthropic_missing_credentials_builds_error_with_hint_when_openai_key_is_set() {
        // given
        let _lock = env_lock();
        let _openai = EnvVarGuard::set("OPENAI_API_KEY", Some("sk-openrouter-varleg"));
        let _xai = EnvVarGuard::set("XAI_API_KEY", None);
        let _dashscope = EnvVarGuard::set("DASHSCOPE_API_KEY", None);

        // when
        let error = anthropic_missing_credentials();

        // then
        match &error {
            ApiError::MissingCredentials {
                provider,
                env_vars,
                hint,
            } => {
                assert_eq!(*provider, "Anthropic");
                assert_eq!(*env_vars, &["ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_API_KEY"]);
                let hint_value = hint.as_deref().expect("hint should be populated");
                assert!(
                    hint_value.contains("OPENAI_API_KEY is set"),
                    "hint should name the detected env var: {hint_value}"
                );
            }
            other => panic!("expected MissingCredentials variant, got {other:?}"),
        }
        let rendered = error.to_string();
        assert!(
            rendered.starts_with("missing Anthropic credentials;"),
            "canonical base message should still lead the rendered error: {rendered}"
        );
        assert!(
            rendered.contains(" — hint: I see OPENAI_API_KEY is set"),
            "rendered error should carry the env-driven hint: {rendered}"
        );
    }

    #[test]
    fn anthropic_missing_credentials_hint_ignores_empty_string_values() {
        // given
        let _lock = env_lock();
        // An empty value is semantically equivalent to "not set" for the
        // credential discovery path, so the sniffer must treat it that way
        // to avoid false-positive hints for users who intentionally cleared
        // a stale export with `OPENAI_API_KEY=`.
        let _openai = EnvVarGuard::set("OPENAI_API_KEY", Some(""));
        let _xai = EnvVarGuard::set("XAI_API_KEY", None);
        let _dashscope = EnvVarGuard::set("DASHSCOPE_API_KEY", None);

        // when
        let hint = anthropic_missing_credentials_hint();

        // then
        assert!(
            hint.is_none(),
            "empty env var should not trigger the hint sniffer, got {hint:?}"
        );
    }

    #[test]
    fn openai_base_url_overrides_anthropic_fallback_for_unknown_model() {
        // given — user has OPENAI_BASE_URL + OPENAI_API_KEY but no Anthropic
        // creds, and a model name with no recognized prefix.
        let _lock = env_lock();
        let _base_url = EnvVarGuard::set("OPENAI_BASE_URL", Some("http://127.0.0.1:11434/v1"));
        let _api_key = EnvVarGuard::set("OPENAI_API_KEY", Some("dummy"));
        let _anthropic_key = EnvVarGuard::set("ANTHROPIC_API_KEY", None);
        let _anthropic_token = EnvVarGuard::set("ANTHROPIC_AUTH_TOKEN", None);

        // when
        let provider = detect_provider_kind("qwen2.5-coder:7b");

        // then — should route to OpenAI, not Anthropic
        assert_eq!(
            provider,
            ProviderKind::OpenAi,
            "OPENAI_BASE_URL should win over Anthropic fallback for unknown models"
        );
    }

    // NOTE: a "OPENAI_BASE_URL without OPENAI_API_KEY" test is omitted
    // because workspace-parallel test binaries can race on process env
    // (env_lock only protects within a single binary). The detection logic
    // is covered: OPENAI_BASE_URL alone routes to OpenAi as a last-resort
    // fallback in detect_provider_kind().
}
