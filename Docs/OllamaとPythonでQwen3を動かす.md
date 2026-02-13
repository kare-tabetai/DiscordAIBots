# **Qwen3アーキテクチャの全貌とOllamaエコシステムにおけるPython統合の技術的詳解**

ローカル環境における大規模言語モデル（LLM）の運用は、データ機密性の保持、レイテンシの極小化、および計算コストの予測可能性という観点から、現代のエンタープライズAI戦略において不可欠な要素となっている。アリババ・グループのQwenチームが提供するQwen3シリーズは、その洗練されたアーキテクチャとオープンウェイトモデルとしての高い汎用性により、ローカルLLMの標準を再定義する存在となった 1。本報告書では、Ollamaプラットフォームを基盤としたQwen3の展開、Python SDKを用いた高度なシステム統合、および計算効率を最大化するためのハードウェア最適化戦略について、専門的知見に基づき包括的に論じる。

## **次世代ローカルLLMとしてのQwen3のパラダイムシフト**

Qwen3は、従来の高密度（Dense）モデルの枠組みを超え、混合エキスパート（Mixture-of-Experts, MoE）技術と、明示的な「思考プロセス」を分離したハイブリッド推論エンジンを導入したことで、オープンソースAIの到達点を一段階引き上げた 1。Qwen3ファミリーは、0.6Bから235Bに至る広範なパラメータスケールを展開しており、モバイルデバイスからハイエンドのGPUクラスターまで、あらゆる計算環境に対応する柔軟性を備えている 2。

### **Qwen3ファミリーの階層的構造と計算資源の最適配分**

Qwen3の設計思想は、特定のタスクに最適なモデルサイズを柔軟に選択できる階層性に反映されている。各モデルは、特定のレイヤー数、アテンション・ヘッドの構成、および埋め込みベクトルの共有設定（Tie Word Embeddings）を最適化することで、推論時のスループットと精度の均衡を保っている 2。

| モデル名称 | 総パラメータ数 | アクティブパラメータ数 | レイヤー数 | コンテキスト長 | 主要アーキテクチャ特性 |
| :---- | :---- | :---- | :---- | :---- | :---- |
| Qwen3-0.6B | 0.6B | 0.6B | 28 | 32,768 | 軽量エッジデバイス最適化、Tie Embedding採用 2 |
| Qwen3-1.7B | 1.7B | 1.7B | 28 | 32,768 | モバイル・ウェアラブル端末向け、高効率推論 2 |
| Qwen3-4B | 4.0B | 4.0B | 36 | 32,768 | 開発者向け標準モデル、GQA（Grouped Query Attention）採用 2 |
| Qwen3-8B | 8.2B | 8.2B | 36 | 131,072 | 多言語・汎用推論、128Kコンテキスト対応 2 |
| Qwen3-14B | 14.8B | 14.8B | 40 | 131,072 | 高度な論理推論、GQA 40/8ヘッド構成 2 |
| Qwen3-32B | 32.8B | 32.8B | 64 | 131,072 | 複雑な命令追従、フラグシップ・デンスモデル 2 |
| Qwen3-30B-A3B | 30.5B | 3.3B | 48 | 131,072 | 高効率MoE、アクティブ3Bによる高速応答 2 |
| Qwen3-235B-A22B | 235B | 22B | 94 | 262,144 | 研究・エンタープライズ級、最高峰の推論性能 1 |

特に、Qwen3-30B-A3Bに代表されるMoEアーキテクチャは、推論に必要な計算量をアクティブなパラメータ数（約3.3B）に限定しつつ、膨大な知識ベースを維持することを可能にしている 2。これは、従来のQwQ-32Bと比較して、約10分の1のアクティブパラメータ数でありながら、同等以上の推論精度を達成するという劇的な効率化を実現している 1。

### **2507アップデートによる長文読解能力の飛躍**

2025年7月に発表された「Qwen3-2507」シリーズは、コンテキスト管理において革命的な進歩をもたらした 3。このアップデートでは、Qwen3-Instruct-2507（非思考モード）とQwen3-Thinking-2507（思考モード）の二分化が進む一方で、最大100万トークンという極大コンテキストの入力を処理する能力が追加された 3。これにより、学術論文の全編解析、ソースコード全体の依存関係把握、および大規模な法律文書の検索・要約といった、従来のLLMでは分割処理が必要だったタスクがシームレスに実行可能となっている 3。

## **Ollamaサーバーの内部構造と環境制御の高度化**

Ollamaは、ローカル環境におけるLLMの展開を抽象化し、一貫したAPIインターフェースを提供するミドルウェアである 12。macOS、Windows、およびLinuxという主要なOSをカバーし、背後でllama.cppを用いた高度なバックエンド最適化を行っている 14。

### **プラットフォーム固有の導入要件と依存関係**

Ollamaのインストールプロセスは、各OSの低レイヤーな計算リソース管理と密接に関連している。

1. **Linuxシステムにおける展開**: curl \-fsSL https://ollama.com/install.sh | sh という単純なコマンドにより、systemdサービスとしての登録までが自動化される 14。Linux環境では、AMD GPUユーザー向けにROCmドライバを同梱した専用パッケージ（ollama-linux-amd64-rocm.tgz）も提供されており、NVIDIA一辺倒ではない計算環境の構築をサポートしている 14。  
2. **Windows環境における構成**: Windows 10 22H2以降、あるいはWindows 11を対象とし、NVIDIAドライバ 452.39以降が推奨される 16。インストールプログラムはユーザープロファイル内にバイナリを配置し、管理者権限なしでの実行を可能にしている 16。  
3. **macOS環境における最適化**: Apple Silicon（M1, M2, M3, M4）のユニファイドメモリを最大限に活用し、Metalフレームワークを介したGPU加速を行う 15。macOS Sonoma（v14）以降がシステム要件として設定されており、CLIバイナリは /usr/local/bin/ollama を通じてアクセス可能となる 15。

### **環境変数によるサーバー挙動の微調整**

Ollamaの動作は、複数の環境変数を設定することで、特定のハードウェア環境やネットワークトポロジーに最適化できる 20。

* **OLLAMA\_HOST**: デフォルトの 127.0.0.1:11434 を 0.0.0.0:11434 に変更することで、ローカルネットワーク上の他のデバイスからのAPIリクエストを受信可能にする 20。  
* **OLLAMA\_MODELS**: モデルの保存先を高速なNVMe SSDやネットワークストレージ（NAS）に変更する際に使用される 16。  
* **OLLAMA\_KEEP\_ALIVE**: モデルをメモリに保持する期間を定義する。デフォルトは5分だが、-1 を設定することで、メモリが許す限りモデルを常駐させ、初回ロードの遅延を回避できる 20。  
* **OLLAMA\_MAX\_LOADED\_MODELS**: 複数のGPUを搭載したシステムにおいて、並列でロード可能なモデルの最大数を制限し、VRAMの競合を防止する 20。  
* **OLLAMA\_KV\_CACHE\_TYPE**: KVキャッシュの量子化タイプ（デフォルトは f16）を q8\_0 や q4\_0 に設定することで、長文生成時のメモリ消費を抑制する 18。

## **Python SDKを通じた推論パイプラインの構築と非同期処理**

Ollamaの真価は、公式のPython SDKを通じて既存のソフトウェアスタックに統合された際に発揮される 12。このライブラリは、REST APIを抽象化し、型定義されたレスポンスオブジェクトと非同期実行モデルを提供する 24。

### **同期・非同期チャットの実装とエラーハンドリング**

標準的な ollama.chat メソッドは、メッセージのリストを受け取り、モデルからの応答を同期的に待機する 24。一方で、高スループットが求められるWebサーバーや非同期エージェントでは、AsyncClient の活用が推奨される。

Python

import asyncio  
from ollama import AsyncClient

async def execute\_qwen3\_inference():  
    client \= AsyncClient()  
    message \= {'role': 'user', 'content': 'Qwen3のアーキテクチャについて技術的に解説してください。'}  
      
    try:  
        \# 非同期でのチャットリクエスト  
        response \= await client.chat(  
            model='qwen3:8b',  
            messages=\[message\],  
            options={'temperature': 0.7, 'top\_p': 0.9}  
        )  
        print(f"Response: {response.message.content}")  
    except Exception as e:  
        print(f"Inference error: {str(e)}")

asyncio.run(execute\_qwen3\_inference())

この実装において、options パラメータはモデルの生成挙動を制御する重要な役割を果たす 24。temperature（温度パラメータ）は出力の多様性を調整し、top\_p（核サンプリング）は確率分布の累積閾値を設定することで、不自然なトークンの選択を抑制する 24。

### **生成パラメータの精緻な制御**

Ollamaの options 辞書、あるいは Options クラスを通じて設定可能なパラメータは多岐にわたり、これらは llama.cpp のバックエンドパラメータに直接マップされる 24。

| パラメータ名 | データ型 | 推奨値 (思考モード) | 推奨値 (非思考モード) | 説明 |
| :---- | :---- | :---- | :---- | :---- |
| temperature | float | 0.6 | 0.7 | 生成のランダム性を制御。思考モードでは低めに設定して論理的一貫性を優先 11 |
| top\_p | float | 0.95 | 0.8 | 累積確率に基づくサンプリング。思考モードでは広い探索範囲を許容 11 |
| top\_k | int | 20 | 20 | 各ステップで考慮する上位トークン数。Qwen3では20が標準 11 |
| num\_ctx | int | 32,768 | 32,768 | コンテキストウィンドウのサイズ。ハードウェアVRAMに応じて調整 5 |
| repeat\_penalty | float | 1.1 | 1.2 | 同じ単語の繰り返しを抑制。1.0〜2.0の範囲で調整 5 |
| seed | int | 42 | 42 | 再現性のための乱数シード。デバッグやベンチマークに不可欠 24 |

## **ハイブリッド推論：思考モード（Thinking Mode）の深層的理解と実装**

Qwen3の最大の特徴は、推論ステップを明示的に出力する「思考モード」の実装にある 26。これは心理学における「二重過程理論」のシステム2、すなわち意識的かつ論理的な思考プロセスをAIに導入したものである 27。

### **思考プロセスのAPIキャプチャ**

Ollamaの最新API（2025年5月以降）では、レスポンスオブジェクトに thinking という専用のフィールドが追加された 27。これにより、推論の軌跡（Reasoning Trace）と最終的な回答をプログラム上で明確に分離して処理することが可能となった。

Python

from ollama import chat

\# thinkパラメータをTrueに設定  
response \= chat(  
    model='qwen3:8b',  
    messages=\[{'role': 'user', 'content': '17 \* 23 をステップバイステップで計算して'}\],  
    think=True,  
    stream=False  
)

\# 思考プロセスの出力  
if hasattr(response.message, 'thinking'):  
    print(f"--- 思考プロセス \---\\n{response.message.thinking}\\n")

\# 最終回答の出力  
print(f"--- 最終回答 \---\\n{response.message.content}")

この分離は、ユーザーインターフェースにおいて「思考中...」というアニメーションを表示したり、複雑な推論を必要とするタスクにおいてのみ推論ログを記録したりといった、高度なUX設計を可能にする 27。

### **タグによる動的モード制御のメカニズム**

Qwen3はプロンプト内に含まれる命令タグを認識し、推論モードを動的に切り替えることができる 2。

* **/think タグ**: モデルに深層的な推論を強制する。数学的な証明や難解なコードのデバッグにおいて威力を発揮する 26。  
* **/no\_think タグ**: 推論ステップを省略し、即座に結論を出力させる。定型的な質問や速度が優先されるチャットボット用途に適している 26。

これらのタグは、システムプロンプトまたはユーザーメッセージの冒頭に配置することで、マルチターンの会話中であってもその都度挙動を変更することが可能である 25。ただし、一部のOllamaのバージョンアップに伴い、プロンプト内命令よりもAPIパラメータ（--think=false 等）が優先される仕様変更が行われた経緯があり、確実な制御のためにはAPI側での明示的な設定が推奨される 32。

## **計算効率の極限：量子化・キャッシュ・アテンションの最適化**

ローカル環境でのLLM運用において最大のボトルネックとなるのはGPUのVRAM容量である。Qwen3を効率的に動作させるためには、単なるハードウェアの増強ではなく、ソフトウェアレベルでの最適化技術の深い理解が求められる。

### **量子化技術によるVRAMフットプリントの削減**

量子化とは、モデルの重みを表現する数値の精度（通常は16ビット浮動小数点数）を、より少ないビット数（8ビットや4ビット）に圧縮する技術である 5。

| 量子化形式 | VRAM削減率 | 推論精度への影響 | 推奨される用途 |
| :---- | :---- | :---- | :---- |
| FP16 (16-bit) | 0% (基準) | なし | 最高精度が必要な研究用途、十分なVRAMがある環境 5 |
| INT8 (8-bit) | 約50%減 | 極めて軽微 | 汎用的な利用、バランス重視の環境 5 |
| Q4\_K\_M (4-bit) | 約75%減 | わずかに知覚可能 | 消費者向けGPUでの標準、14B〜32Bモデルの実行 5 |
| Q3\_K\_XL (3-bit) | 約80%減 | 明確な知能低下の可能性あり | 8B以上のモデルを極端にメモリ制限がある環境で動かす場合 26 |

例えば、Qwen3-14BをFP16で動かすには約30GB以上のVRAMが必要となるが、4ビット量子化（Q4\_K\_M）を適用することで、一般的なRTX 3060（12GB）やRTX 4070（12GB）といった、より安価なGPUでも動作可能となる 5。

### **フラッシュ・アテンションとスループットの向上**

アテンション機構の計算量は入力トークン数の二乗に比例する性質を持つが、Flash Attentionを有効化することで、このボトルネックを緩和できる 19。

* **Flash Attentionの数理的意義**: 標準的なアテンション計算が、行列演算の結果を一度メインメモリ（VRAM）に書き出すのに対し、Flash AttentionはGPU内の高速なSRAMを活用してタイリングを行い、メモリアクセスのオーバーヘッドを劇的に削減する 33。  
* **Ollamaでの有効化**: export OLLAMA\_FLASH\_ATTENTION=1 を設定することで、Qwen3の長文推論における生成速度（Tokens per second）が大幅に改善される 18。ベンチマークによれば、Qwen3-4BをRTX 3060で動作させた場合、小規模なコンテキストでは約110 tok/s、スループット重視の設定では最大400 tok/sに近い速度を記録している 18。

## **構造化出力とツール・コーリングによる自律的エージェントの実現**

Qwen3は、単なるテキスト生成器を超えて、外部システムと連携する「エージェント」としての高い適性を備えている 1。

### **Pydanticを用いた厳密なデータ抽出**

LLMの出力をプログラムで確実に扱うためには、JSONモードとスキーマ検証の組み合わせが不可欠である 34。

Python

from ollama import chat  
from pydantic import BaseModel, Field

class EntityExtraction(BaseModel):  
    person\_names: list\[str\] \= Field(description="抽出された人物名")  
    locations: list\[str\] \= Field(description="抽出された地名")  
    sentiment: str \= Field(description="文章の感情分析結果")

\# JSONスキーマをOllamaに渡す  
response \= chat(  
    model='qwen3:8b',  
    messages=\[{'role': 'user', 'content': '昨日、佐藤さんは東京から大阪へ向かいました。'}\],  
    format\=EntityExtraction.model\_json\_schema()  
)

\# Pydanticによる自動バリデーション  
data \= EntityExtraction.model\_validate\_json(response.message.content)

このアプローチにより、LLMが生成する自然言語の中に埋もれた情報を、型安全なPythonオブジェクトとして取り出すことが可能となる 36。

### **関数呼び出し（Tool Calling）のオーケストレーション**

Qwen3は関数の仕様（Docstring）を理解し、適切な引数で呼び出すべきタイミングを判断できる 35。

* **ツールの定義**: Pythonの関数を定義し、そのメタデータをOllamaに渡すことで、モデルは「このタスクにはあのツールが必要だ」と判断し、関数の名前と引数を含むJSONを返却する 35。  
* **思考モードとの融合**: think=True を設定した状態でツール・コーリングを行うと、モデルは「なぜこのツールを使うのか」を思考プロセスで自己批判的に検討した上で、最適な関数呼び出しを行うため、誤作動の確率が低下する 35。

## **ローカル環境におけるVRAM管理と最適ハードウェア選定**

Qwen3の各モデルを運用するためのハードウェア構成は、単に「動く」ことと「快適に動く」ことの間に大きな隔たりがある。

### **ハードウェア構成と推論速度の相関**

以下の表は、特定のハードウェア構成においてQwen3が達成しうるパフォーマンスの期待値を示している。

| 構成デバイス | 対象モデル | 推論速度 (tok/s) | 備考 |
| :---- | :---- | :---- | :---- |
| Apple M3 Max (128GB) | Qwen3-32B (Q4\_K\_M) | 15〜25 | ユニファイドメモリにより大容量コンテキストでも安定 8 |
| NVIDIA RTX 4090 (24GB) | Qwen3-14B (Q8\_0) | 40〜60 | 8ビット量子化で高精度を維持しつつ高速生成 5 |
| NVIDIA RTX 3060 (12GB) | Qwen3-8B (Q4\_K\_M) | 25〜35 | コストパフォーマンスに優れたエントリー構成 5 |
| Intel CPU (32GB RAM) | Qwen3-4B (Q4\_K\_M) | 2〜5 | GPUなしでも動作可能だが、応答は著しく遅い 19 |

### **AMD ROCm環境における固有の課題**

AMD製GPU（Radeonシリーズ）でのOllama運用は、ROCmフレームワークを通じてサポートされているが、特有の制限も存在する 14。最新のQwen3バリアント（2507など）では、Flash Attentionが正常に機能しない、あるいはKVキャッシュの設定が原因で速度低下（Regression）が発生するとの報告がある 18。安定した運用のためには、特定の環境変数（OLLAMA\_FLASH\_ATTENTION=0 など）の設定が必要になるケースがあり、NVIDIA環境と比較して綿密な動作検証が求められる 18。

## **結論と未来展望：エッジコンピューティングにおける統合AI戦略**

Qwen3とOllama、そしてPythonの組み合わせは、ローカルLLM活用の成熟期における一つの完成形を提示している。思考モードによる論理推論の深化、MoEによる計算効率の最適化、そして100万トークンに及ぶ長文コンテキスト管理能力は、これまでクラウドLLMが独占していた「高度な知的作業」の領域を、個人のローカル環境へと完全に引き寄せた 1。

今後、AIシステムの開発において重要なのは、単一のモデルに依存するのではなく、Qwen3-0.6Bのようなエッジ向けモデルでの一次選別と、Qwen3-235Bのような大規模モデルでの最終推論を組み合わせた階層的な処理系（Tiered Architecture）を構築することである 2。Ollamaが提供する軽量なAPIサーバーと、Pythonの広範なエコシステムを組み合わせることで、開発者はデータプライバシーを妥協することなく、真に自律的で強力なAIアプリケーションを創出することが可能となっている。Qwen3は、単なるツールの提供に留まらず、ローカルAIが人間の知能を拡張する未来の基盤としての役割を、今後も果たし続けるであろう。

#### **引用文献**

1. qwen3 \- Ollama, 2月 13, 2026にアクセス、 [https://ollama.com/library/qwen3](https://ollama.com/library/qwen3)  
2. How to Run Qwen 3 Locally with Ollama & VLLM \- Apidog, 2月 13, 2026にアクセス、 [https://apidog.com/blog/run-qwen-3-locally/](https://apidog.com/blog/run-qwen-3-locally/)  
3. Qwen3 is the large language model series developed by Qwen team, Alibaba Cloud. \- GitHub, 2月 13, 2026にアクセス、 [https://github.com/QwenLM/Qwen3](https://github.com/QwenLM/Qwen3)  
4. Qwen3: Think Deeper, Act Faster | Qwen, 2月 13, 2026にアクセス、 [https://qwenlm.github.io/blog/qwen3/](https://qwenlm.github.io/blog/qwen3/)  
5. How to Run Qwen3 Locally \- A Practical Guide for AI Enthusiasts, 2月 13, 2026にアクセス、 [https://onedollarvps.com/blogs/how-to-run-qwen3-locally](https://onedollarvps.com/blogs/how-to-run-qwen3-locally)  
6. Tags · qwen3 \- Ollama, 2月 13, 2026にアクセス、 [https://ollama.com/library/qwen3/tags](https://ollama.com/library/qwen3/tags)  
7. Qwen 3: What You Need to Know \- Gradient Flow, 2月 13, 2026にアクセス、 [https://gradientflow.com/qwen-3/](https://gradientflow.com/qwen-3/)  
8. Qwen3-14B: Specifications and GPU VRAM Requirements \- ApX Machine Learning, 2月 13, 2026にアクセス、 [https://apxml.com/models/qwen3-14b](https://apxml.com/models/qwen3-14b)  
9. Qwen3-4B: Specifications and GPU VRAM Requirements \- ApX Machine Learning, 2月 13, 2026にアクセス、 [https://apxml.com/models/qwen3-4b](https://apxml.com/models/qwen3-4b)  
10. Which Qwen3 Model Is Right for You? A Practical Guide \- Novita AI Blog, 2月 13, 2026にアクセス、 [https://blogs.novita.ai/which-qwen3-model-is-right-for-you-a-practical-guide/](https://blogs.novita.ai/which-qwen3-model-is-right-for-you-a-practical-guide/)  
11. Quickstart \- Qwen \- Read the Docs, 2月 13, 2026にアクセス、 [https://qwen.readthedocs.io/en/latest/getting\_started/quickstart.html](https://qwen.readthedocs.io/en/latest/getting_started/quickstart.html)  
12. Getting Started with Ollama: Run LLMs on Your Computer | by Jonathan Gastón Löwenstern, 2月 13, 2026にアクセス、 [https://medium.com/@jonigl/getting-started-with-ollama-run-llms-on-your-computer-915ba084918c](https://medium.com/@jonigl/getting-started-with-ollama-run-llms-on-your-computer-915ba084918c)  
13. Ollama's documentation \- Ollama, 2月 13, 2026にアクセス、 [https://docs.ollama.com/](https://docs.ollama.com/)  
14. Ollama Guide: Install & Run LLMs Locally \- centron GmbH, 2月 13, 2026にアクセス、 [https://www.centron.de/en/tutorial/ollama-installation-guide-run-llms-locally-on-linux-windows-macos/](https://www.centron.de/en/tutorial/ollama-installation-guide-run-llms-locally-on-linux-windows-macos/)  
15. macOS \- Ollama's documentation, 2月 13, 2026にアクセス、 [https://docs.ollama.com/macos](https://docs.ollama.com/macos)  
16. Windows \- Ollama's documentation, 2月 13, 2026にアクセス、 [https://docs.ollama.com/windows](https://docs.ollama.com/windows)  
17. Run Qwen3-Coder-Next Locally: Vibe Code an Analytics Dashboard | DataCamp, 2月 13, 2026にアクセス、 [https://www.datacamp.com/tutorial/run-qwen3-coder-next-locally](https://www.datacamp.com/tutorial/run-qwen3-coder-next-locally)  
18. Qwen3 vs Qwen3-2507 Regression caused by flash attention. AMD ROCM · Issue \#12432 · ollama/ollama \- GitHub, 2月 13, 2026にアクセス、 [https://github.com/ollama/ollama/issues/12432](https://github.com/ollama/ollama/issues/12432)  
19. Getting the Most Out of OLLAMA: A Practical Guide | by Alessandro Borges | Medium, 2月 13, 2026にアクセス、 [https://medium.com/@alessandroborges\_84477/getting-the-most-out-of-ollama-a-practical-guide-7ac2dee244ef](https://medium.com/@alessandroborges_84477/getting-the-most-out-of-ollama-a-practical-guide-7ac2dee244ef)  
20. FAQ \- Ollama English Documentation, 2月 13, 2026にアクセス、 [https://ollama.readthedocs.io/en/faq/?h=default+context](https://ollama.readthedocs.io/en/faq/?h=default+context)  
21. FAQ \- Ollama's documentation, 2月 13, 2026にアクセス、 [https://docs.ollama.com/faq](https://docs.ollama.com/faq)  
22. Ollama Installation for macOS, Linux, and Windows \- GitHub Pages, 2月 13, 2026にアクセス、 [https://translucentcomputing.github.io/kubert-assistant-lite/ollama.html](https://translucentcomputing.github.io/kubert-assistant-lite/ollama.html)  
23. docs/faq.md · 9164b0161bcb24e543cba835a8863b80af2c0c21 · Till-Ole Herbst / Ollama \- GitLab, 2月 13, 2026にアクセス、 [https://gitlab.informatik.uni-halle.de/ambcj/ollama/-/blob/9164b0161bcb24e543cba835a8863b80af2c0c21/docs/faq.md](https://gitlab.informatik.uni-halle.de/ambcj/ollama/-/blob/9164b0161bcb24e543cba835a8863b80af2c0c21/docs/faq.md)  
24. ollama/ollama-python: Ollama Python library \- GitHub, 2月 13, 2026にアクセス、 [https://github.com/ollama/ollama-python](https://github.com/ollama/ollama-python)  
25. tryumanshow/Qwen3-8B-no-think \- Hugging Face, 2月 13, 2026にアクセス、 [https://huggingface.co/tryumanshow/Qwen3-8B-no-think](https://huggingface.co/tryumanshow/Qwen3-8B-no-think)  
26. sam860/qwen3:8b-Q4\_K\_M \- Ollama, 2月 13, 2026にアクセス、 [https://ollama.com/sam860/qwen3:8b-Q4\_K\_M](https://ollama.com/sam860/qwen3:8b-Q4_K_M)  
27. Thinking \- Ollama's documentation, 2月 13, 2026にアクセス、 [https://docs.ollama.com/capabilities/thinking](https://docs.ollama.com/capabilities/thinking)  
28. Thinking · Ollama Blog, 2月 13, 2026にアクセス、 [https://ollama.com/blog/thinking](https://ollama.com/blog/thinking)  
29. It's time to turn off the annoying THINK MODE Qwen-3 | by Duke Wang | Medium, 2月 13, 2026にアクセス、 [https://medium.com/@dukewillbe185/its-time-to-turn-off-the-annoying-think-mode-qwen-3-eefb7dedcadd](https://medium.com/@dukewillbe185/its-time-to-turn-off-the-annoying-think-mode-qwen-3-eefb7dedcadd)  
30. Generate a response \- Ollama's documentation, 2月 13, 2026にアクセス、 [https://docs.ollama.com/api/generate](https://docs.ollama.com/api/generate)  
31. Here's how to turn off "thinking" in Qwen 3: add "/no\_think" to your prompt or system message. : r/LocalLLaMA \- Reddit, 2月 13, 2026にアクセス、 [https://www.reddit.com/r/LocalLLaMA/comments/1ka67wo/heres\_how\_to\_turn\_off\_thinking\_in\_qwen\_3\_add\_no/](https://www.reddit.com/r/LocalLLaMA/comments/1ka67wo/heres_how_to_turn_off_thinking_in_qwen_3_add_no/)  
32. Using "/no\_think" with HYBRID models does not work anymore · Issue \#12575 \- GitHub, 2月 13, 2026にアクセス、 [https://github.com/ollama/ollama/issues/12575](https://github.com/ollama/ollama/issues/12575)  
33. How much does flash attention affect intelligence in reasoning models like QwQ \- Reddit, 2月 13, 2026にアクセス、 [https://www.reddit.com/r/LocalLLaMA/comments/1jcsvys/how\_much\_does\_flash\_attention\_affect\_intelligence/](https://www.reddit.com/r/LocalLLaMA/comments/1jcsvys/how_much_does_flash_attention_affect_intelligence/)  
34. Ollama LLM | LlamaIndex Python Documentation, 2月 13, 2026にアクセス、 [https://developers.llamaindex.ai/python/framework/integrations/llm/ollama/](https://developers.llamaindex.ai/python/framework/integrations/llm/ollama/)  
35. Tool calling \- Ollama's documentation, 2月 13, 2026にアクセス、 [https://docs.ollama.com/capabilities/tool-calling](https://docs.ollama.com/capabilities/tool-calling)  
36. Structured Outputs \- Ollama's documentation, 2月 13, 2026にアクセス、 [https://docs.ollama.com/capabilities/structured-outputs](https://docs.ollama.com/capabilities/structured-outputs)