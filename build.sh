#!/bin/bash
# Script build cho Linux
echo "🚀 Đang bắt đầu build Claw Code (Release)..."
cd rust
cargo build --release --workspace
echo "✅ Build hoàn tất! File thực thi nằm tại: rust/target/release/claw"
