# DiscordAIBots

Discord上で送信されたGoogleドライブの画像URLを、OllamaとQwen3 4Bモデルで評価し、結果を返信するBotプログラム。

## プロジェクト構成

```
DiscordAIBots/
├── src/                    # ソースコード
├── Docs/                   # 技術ドキュメント
├── .env                    # 環境変数（Discord Token等）
├── pyproject.toml          # プロジェクト設定
└── CLAUDE.md               # このファイル
```

## 技術スタック

- **Python**: 3.11以上
- **Discord Bot**: discord.py
- **AI**: Ollama + Qwen3 4B it
- **画像取得**: aiohttp（Google Drive API）

## 開発ルール

### Python環境
グローバル環境を汚さないため、以下のいずれかを使用:

```bash
# uvを使う場合（推奨）
uv venv
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate     # Windows

# venvを使う場合
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate     # Windows
```

### パッケージインストール
```bash
# uvの場合
uv pip install discord.py ollama aiohttp python-dotenv

# pipの場合
pip install discord.py ollama aiohttp python-dotenv
```

### Git操作
ghコマンドを使用してブランチ作成、コミット、プルリクエストを行う:

```bash
# ブランチ作成
git checkout -b feature/機能名

# コミット
git add .
git commit -m "コミットメッセージ"

# PRの作成
gh pr create --title "タイトル" --body "説明"
```

### ドキュメント
人間の操作が必要な部分（Discord Bot設定、Ollamaセットアップ等）は `Docs/` にマークダウンで保存する。

## 技術参照

関連する技術情報は以下のドキュメントを参照:

- `Docs/Discord BotでPython処理を呼び出す.md` - Discord.pyの使い方、非同期処理
- `Docs/OllamaとPythonでQwen3を動かす.md` - Qwen3モデルの設定、Python SDK

## 必要なセットアップ（人間の操作）

1. Discord Developer Portalでアプリ作成・Bot Token取得
2. Ollamaのインストールと`qwen3:4b`モデルのプル
3. Google Cloud ConsoleでAPI有効化（必要に応じて）
4. `.env`ファイルの作成

詳細は `Docs/` 内のドキュメントを参照。
