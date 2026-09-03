# Validation Log — รันจริงบนข้อมูลจริง

เอกสารนี้บันทึกสิ่งที่ **รันจริงแล้ว** ไม่ใช่สิ่งที่คิดว่าน่าจะทำงาน
ทุกตัวเลขในนี้มาจากการรันในเซสชันวันที่ 2026-09-03

---

## ข้อมูลที่ใช้

| ชุด | ที่มา | ช่วง | บาร์ |
|---|---|---|---|
| M5 | `GC=F` (COMEX gold futures) ผ่าน Yahoo | 2026-06-25 → **2026-09-03 (วันนี้)** | 13,755 |
| H1 | `GC=F` เหมือนกัน | 2024-04-12 → **2026-09-03** (2 ปี 5 เดือน) | 13,742 |

ดึงด้วย `tools/fetch_public_gold.py` มี session gap จริง 50 และ 139 ช่วงตามลำดับ

> ⚠️ **ข้อมูลชุดนี้ใช้ทดสอบโค้ด ไม่ใช่ใช้เทรน model ที่จะเทรดจริง**
> GC=F คือ futures ไม่ใช่ XAUUSD spot, timestamp เป็น UTC ไม่ใช่เวลาเซิร์ฟเวอร์โบรกเกอร์,
> volume เป็น contract volume ไม่ใช่ tick volume — ทั้งสามข้อทำให้ model ที่ได้ใช้เทรดไม่ได้
> ของจริงต้องใช้ `tools/fetch_mt5_history.py`

---

## ✅ ผ่าน

### Feature parity บนข้อมูลจริง
```
M5 : worst relative difference = 1.784e-09   (close_zscore_20)
H1 : worst relative difference = 2.491e-10   (close_zscore_20)
```
ทนต่อ weekend gap, session break, บาร์ที่ high == low, บาร์ volume = 0

ทดสอบ negative case ด้วย: สลับ `rsi_7` ↔ `rsi_14` ใน `Features.mqh` → harness ฟ้องทั้งสอง index และ FAIL

### Full chain บนบาร์ล่าสุดจริง
```
bar 2026-09-03 22:00  close=4522.60
  MQL5-transcription vs pandas : max |diff| = 0.0
  ONNX primary  P(up) = 0.4889  -> SELL
  ONNX meta     conf  = 0.5717  -> REFUSE (gate 0.60)
```
คือเส้นทางเดียวกับที่ EA จะเดินทุกแท่ง

### Export ปฏิเสธของเสียจริง
- คู่ RandomForest → flip rate 0.000% ทั้งสองชั้น → เขียนไฟล์ 4 ไฟล์
- คู่ HistGradientBoosting → flip rate 3.202% / 0.432% → **ไม่เขียนอะไรเลย**

---

## 🔴 บั๊กที่เจอเพราะรันจริง (ไม่มีทางเจอด้วยการอ่านโค้ด)

### 1. `HistGradientBoostingClassifier` แปลงเป็น ONNX แล้วเพี้ยน

ตัวตรวจเดิมของผมใช้ **random vector** แล้วรายงาน drift `1.3e-07` → ผ่านฉลุย
พอเปลี่ยนมาใช้ **feature vector จริง 12,729 แถว**:

```
                    |onnx - sklearn|                 decision flips
model        median      p99        max             ที่ gate 0.60
hgb        3.02e-08   5.42e-02   1.71e-01              1.752%
gb         1.78e-08   6.35e-08   1.10e-07              0.000%
rf         2.48e-08   1.03e-07   1.71e-07              0.000%
```

**1.75% ของการตัดสินใจเปลี่ยนไป** ระหว่าง `.pkl` ที่คุณ validate กับ `.onnx` ที่ EA รันจริง
สาเหตุ: HGB แบ่ง feature เป็น bin ด้วยขอบแบบ float64 ซึ่งไม่รอดตอนแปลงเป็น float32 ใน ONNX
ค่าที่อยู่ใกล้ขอบ bin จะวิ่งไปคนละกิ่ง

random vector มองไม่เห็นเพราะมันอยู่ไกลจาก distribution ที่เทรน
ทุก tree เลยวิ่งไป leaf สุดขอบเหมือนกันหมด ไม่ว่าจะปัดเศษยังไง

**แก้แล้ว:**
- default model เปลี่ยนเป็น `RandomForest` (แม่นเป๊ะ และเร็วกว่า `gb` 15 เท่า)
- `export_to_onnx.py` มี `--verify-csv` ตรวจบน vector จริง และรายงาน **flip rate** ไม่ใช่แค่ drift
- ถ้า flip rate > 0.1% มัน **ปฏิเสธ ไม่เขียนไฟล์**

ตารางเวลาที่วัดได้ (10,183 แถว × 40 features):

| model | fit time | vs hgb | test AUC | ONNX flips |
|---|---|---|---|---|
| hgb | 0.6 s | 1.0× | 0.5062 | ❌ 1.75% |
| gb | 33.7 s | 56.6× | 0.5093 | ✅ 0.000% |
| **rf** | **2.3 s** | **3.9×** | **0.5198** | ✅ **0.000%** |

### 2. Exporter เขียนไฟล์ทีละตัว → เหลือคู่ที่ไม่แมตช์กัน

ตอน primary ถูกปฏิเสธแต่ meta ผ่าน มันเขียน `meta.onnx` ทิ้งไว้
บวกกับ `primary.onnx` เก่าจากรันก่อนหน้า = **สองชั้นคนละรุ่นคุยกัน**

**แก้แล้ว:** แปลงและตรวจทั้งคู่ก่อน แล้วค่อยเขียนพร้อมกัน หรือไม่เขียนเลย

### 3. Meta tier ถูกตรวจด้วย vector ปลอม

`meta.onnx` รับ 43 มิติ ซึ่ง 3 ตัวท้ายมาจาก primary
ตอนแรกถ้า primary export ล้ม ตัวตรวจ meta จะถอยไปใช้ random vector แล้วผ่านหน้าตาเฉย

**แก้แล้ว:** สร้าง meta vector จาก **primary pickle** เสมอ ไม่เกี่ยวกับว่า export ผ่านไหม

---

## 🔴 สิ่งที่อันตรายที่สุดที่พบ: ตัวเลขที่ดูดีแต่ไม่มีความหมาย

รันบน H1 จริง 2 ปี ผลที่ออกมาตอนแรก:

```
FINAL HOLDOUT   thr=0.50   trades=1374   PF=1.169   E[r]=+0.000532
```

PF 1.17 บนข้อมูลที่ไม่เคยแตะ 1,374 ไม้ — **ใครก็ตามที่เห็นตัวเลขนี้จะเอาไปเทรด**

แต่ label แบบ triple-barrier ที่ horizon 12 แท่ง ทำให้แต่ละไม้ **ซ้อนทับกับ 12 ไม้ข้างเคียง**
1,374 ไม้ ไม่ใช่ 1,374 ตัวอย่างอิสระ มันคือประมาณ **114**

```
t (แบบที่ทุกคนคำนวณ)          = 2.54    ← ดูมีนัยสำคัญ p<0.05
t (ปรับตาม label overlap)      = 0.73    ← ไม่มีอะไรเลย
block-bootstrap 95% CI ของ E[r] = [-0.000158, +0.001152]   ← คร่อมศูนย์
P(E[r] <= 0)                   = 0.075
```

และยังไม่นับว่า threshold sweep ลอง 12 ค่าแล้วเลือกอันที่ดีที่สุด → selection bias อีกชั้น

**เพิ่มเข้าไปแล้วใน `train_two_tier.py`:**
- คอลัมน์ `independent~` (= trades ÷ horizon) ในทุกตาราง
- t-stat แบบปรับ overlap
- moving-block bootstrap (block = horizon) → CI 95% และ P(E[r] ≤ 0)
- Šidák correction สำหรับจำนวน threshold ที่ลอง
- **คำตัดสิน PASS / FAIL ต่อ threshold** พร้อมเหตุผล

ผลหลังใส่เข้าไป — ทั้ง M5 และ H1 บนข้อมูลจริง:

```
NO THRESHOLD CLEARS THE BAR.
```

ซึ่ง**ถูกต้องแล้ว** indicator ทั่วไป 40 ตัวบนทองไม่ควรมี edge
เครื่องมือที่บอกว่า "ไม่มี" ตอนที่ไม่มีจริง คือเครื่องมือที่เชื่อได้ตอนมันบอกว่า "มี"

---

## ⚠️ ยังไม่ได้ตรวจ

| รายการ | สถานะ |
|---|---|
| **คอมไพล์ MQL5** | ❌ ทำไม่ได้ในสภาพแวดล้อมนี้ ตรวจได้แค่ balance/include/ลำดับฟีเจอร์/API signature |
| **`OnnxRun` ใน MT5 จริง** | ❌ ตรวจแค่ฝั่ง onnxruntime ว่า signature ตรง (`label` int64[N] + `probabilities` float[N,2]) |
| Parity ด่านที่ 3 (MT5 ↔ Python) | ❌ ต้องรัน `ParityCheck.mq5` ในเทอร์มินัล |
| Strategy Tester | ❌ |
| การส่งออเดอร์จริง / SL/TP / trailing | ❌ ไม่มีทางตรวจนอกเทอร์มินัล |
| Panel บนชาร์ต | ❌ |
| ข้อมูล XAUUSD spot เวลาเซิร์ฟเวอร์จริง | ❌ ต้องรัน `fetch_mt5_history.py` บนเครื่อง Windows |

**สรุป: ฝั่ง Python ทดสอบแล้วบนข้อมูลจริง ฝั่ง MQL5 ยังไม่เคยผ่านคอมไพเลอร์แม้แต่ครั้งเดียว**
