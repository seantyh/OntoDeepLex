Purposes
--------
* P1 Sense Tagging
* P2 CWN Expansion
* P3 CWN/PWN Alignment

Annotation Tasks
----------------
* T1 WSD Check  
  * T1.1 用CWN2019(`A0.1`)當作資料來源，先訓練出WSD baseline model(`A1.1`)
  * T1.2 從ASBC5(`A0.2`)中抽取出$X_{sentence}$ 個句子。
  * T1.3 用baseline model(`A1.1`)對前步產生句子做詞意消岐，並將結果整理成待修正的pre-tagged data(`A2.1`)
  * T1.4 標記者用WSD修訂介面（`M3`）完成標記修訂，修訂結果為human-editted data(`A2.2`)

* T2 CWN Expansion  
  * T2.1 新增詞彙
    > 那些在WSD中沒cover到的詞彙，如果符合下列條件，則應該加入CWN：
    > 1. 高頻/keyness(tf-idf)  
    > 2. N or V
    > 3. 在華語8K詞
    > 4. 非專有名詞

    > 新增的詞彙必須有下列訊息
    > 1. 能做詞意區辨(sense distinction):
    >    a. 定義寫得出來
    >    b. 要有例句（>= 3）
    > 2. 能定義在CWN的relations，
    >    最好能接上PWN。
    
    > CWN/PWN alignment
    > * mapping_synonym
    > * mapping_hypernym
    > * mapping_holonym_(member/part/substance)

  * T2.2 檢查/建立CWN原有sense和PWN的關係 

Models
------
* M1 Sense Tagger
    * M1.0 Sense tagger - baseline
    * M1.1 Sense tagger - next
* M2 Front-end
    * M2.1 Static webpage
    * M2.2 Visualization (CWN)
    * M2.3 WSD demo
    * M2.4 CWN editor    
* M3 Sense revision interface

Artifacts
----------
* A0 Existing Resources
    * A0.1 CWN-2019
    * A0.2 ASBC5
* A1 WSD Models
    * A1.1 Baseline WSD
    * A1.2 WSD next
* A2 WSD annotation manual check
    * A2.1 pre-tagged sense data
    * A2.2 human-editted sense data
* A3 CWN additional entries
* A4 CWN/PWN alignments
    * A4.1 CWN additional alignments
    * A4.2 CWN alignment checks
* A5 CWN-2020
