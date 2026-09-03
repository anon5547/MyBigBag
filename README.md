# Sniper AI V5 — MT5 Expert Advisor (Two-Tier Meta-Labelling)

EA เนทีฟ MQL5 ที่เขียนใหม่ทั้งหมดจากบอท Python **"Sniper AI Control Panel V4"**
โครงสร้าง AI เหมือนเดิม (Primary + Meta) แต่ย้ายทุกอย่างเข้าไปรันในเทอร์มินัล MT5

| | V4 (Python) | V5 (MQL5) |
|---|---|---|
| Inference | Python + `.pkl` | ONNX native ในเทอร์มินัล |
| Trailing / SL | ลูปใน Python — เน็ตหลุด = ออเดอร์ลอย | **SL/TP อยู่ที่เซิร์ฟเวอร์โบรกเกอร์** |
| Backtest | ไม่มี | Strategy Tester เต็มรูปแบบ |
| GUI | tkinter (แยกโปรเซส, ค้างได้) | Panel บนชาร์ต |
| Feature parity | ไม่มีการตรวจ | มี test บังคับ (`tools/parity_check.py`) |

> ⚠️ **อ่าน [`docs/REVIEW.md`](docs/REVIEW.md) ก่อนแตะบัญชีเงินจริง** — ในนั้นคือจุดตายของระบบ V4 ที่เห็นจากสกรีนช็อตของคุณ ทั้งหมด ไม่มีอวย

---

## 1. โครงสร้างไฟล์

```
MQL5/
  Experts/SniperAI/SniperAI_V5.mq5     ← EA หลัก
  Scripts/SniperAI/ParityCheck.mq5     ← สคริปต์พิสูจน์ว่าฟีเจอร์ live == ฟีเจอร์ตอนเทรน
  Include/SniperAI/
    MathUtil.mqh      ← EMA / RMA / RSI / ADX / ATR (deterministic)
    Features.mqh      ← ฟีเจอร์ 40 มิติ
    Inference.mqh     ← ONNX 2 ชั้น (primary + meta gate)
    Broker.mqh        ← lot/stops/filling/retry ทั้งหมดที่โบรกเกอร์ใช้ปฏิเสธออเดอร์
    RiskManager.mqh   ← money/points/ATR stops, trailing, break-even, ลิมิตรายวัน
    Filters.mqh       ← session / blackout / spread / regime
    Panel.mqh         ← หน้า UI บนชาร์ต
    Logger.mqh        ← journal + ring buffer + CSV audit
tools/
  feature_contract.json   ← สัญญาฟีเจอร์ (แหล่งความจริงเดียว)
  sniper_features.py      ← ฝาแฝด Python ของ Features.mqh
  mql5_transcription.py   ← ถอดโค้ด MQL5 เป็น Python ตรงตัว เพื่อใช้เทียบ
  parity_check.py         ← ตัวตรวจ parity (ต้องผ่านก่อนเทรน)
  train_two_tier.py       ← เทรน primary + meta (triple barrier, purged CV)
  export_to_onnx.py       ← .pkl → .onnx ที่ MQL5 อ่านได้
  fetch_mt5_history.py    ← ดึงบาร์จาก MT5 เป็น "เวลาเซิร์ฟเวอร์"
docs/
  ARCHITECTURE.md   REVIEW.md
```

---

## 2. ติดตั้ง

1. เปิด MetaTrader 5 → **File → Open Data Folder**
2. ก๊อป `MQL5/Experts/SniperAI` → `MQL5/Experts/SniperAI`
3. ก๊อป `MQL5/Include/SniperAI` → `MQL5/Include/SniperAI`
4. ก๊อป `MQL5/Scripts/SniperAI` → `MQL5/Scripts/SniperAI`
5. สร้างโฟลเดอร์ `MQL5/Files/SniperAI` (ที่เก็บโมเดล `.onnx`)
6. เปิด MetaEditor → กด **Compile** ที่ `SniperAI_V5.mq5`

**ต้องใช้ MetaTrader 5 build 3620 ขึ้นไป** (ONNX runtime ในตัว) — EA จะบอกเองถ้าเก่าเกิน

ฝั่ง Python (เครื่องไหนก็ได้ ไม่ต้องเป็นเครื่องเดียวกับ MT5):
```bash
pip install -r tools/requirements.txt
```

---

## 3. ลำดับงานที่ถูกต้อง (ห้ามข้ามขั้น)

### ขั้น 0 — ดึงข้อมูล "เวลาเซิร์ฟเวอร์"
```bash
python tools/fetch_mt5_history.py --symbol XAUUSD --timeframe M5 --bars 200000 --out XAUUSD_M5.csv
```
สคริปต์จะพิมพ์ median spread ออกมาด้วย — จำตัวเลขนั้นไว้ ใช้เป็น `--cost` ตอนเทรน

> ถ้าเทรนด้วยข้อมูล UTC แต่โบรกเกอร์รัน UTC+3 → `hour_sin`/`hour_cos` ผิดทุกแถว
> โมเดลไม่ error แต่ 4 ฟีเจอร์กลายเป็นขยะ และคุณจะไม่มีวันรู้

### ขั้น 1 — ตรวจ parity ก่อน (offline)
```bash
python tools/parity_check.py
```
ต้องขึ้น `PARITY OK` ถ้าไม่ผ่าน ห้ามไปต่อ

### ขั้น 2 — เทรน
```bash
python tools/train_two_tier.py --csv XAUUSD_M5.csv --outdir models/ \
    --horizon 12 --up-mult 1.5 --dn-mult 1.5 --cost 0.30
```
สคริปต์จะพิมพ์ **threshold sweep** ออกมา — ใช้ตารางนั้นเลือก `InpConfThreshold`
ไม่ใช่เดาเอาเอง และไม่ใช่ 0.35 เพราะมันคือ "เอาเกือบทุกสัญญาณ"

### ขั้น 3 — แปลงเป็น ONNX
```bash
python tools/export_to_onnx.py \
    --primary models/primary.pkl --meta models/meta.pkl \
    --outdir "C:/Users/<you>/AppData/Roaming/MetaQuotes/Terminal/<id>/MQL5/Files/SniperAI"
```
ตัว export จะ **ปฏิเสธ** ถ้าโมเดลไม่ตรงสัญญา 40/43 มิติ, ถ้า `classes_` ไม่ใช่ `[0,1]`,
หรือถ้ากราฟ ONNX ออกมาเป็น zipmap (ซึ่ง MQL5 อ่านไม่ได้)

> โมเดลเดิมของคุณ (`XAUUSD_M5_Trinity_Sniper.pkl`, `XAUUSD_M5_Bidirectional_Sniper.pkl`)
> เกือบแน่นอนว่าเทรนด้วยชุดฟีเจอร์คนละชุด — exporter จะบอกความกว้าง input ให้เห็นชัดๆ
> ถ้าไม่ตรง ต้องเทรนใหม่ ไม่ใช่ยัดเข้าไปแล้วหวังว่ารอด

### ขั้น 4 — ปิดลูป parity ระหว่าง MT5 ↔ Python
รันสคริปต์ `ParityCheck` บนชาร์ตที่จะเทรด แล้ว:
```bash
python tools/parity_check.py \
    --csv  ".../MQL5/Files/SniperAI/parity_bars.csv" \
    --mql5 ".../MQL5/Files/SniperAI/parity_mql5.csv"
```
ผ่าน = ฟีเจอร์ตอนเทรนกับตอนเทรดเป็นสิ่งเดียวกันจริง

### ขั้น 5 — Strategy Tester
โหมด **Every tick based on real ticks**, ใส่ commission/spread จริง
ถ้ากราฟ equity สวยเกินไป แปลว่ามีอะไรผิด ไม่ใช่คุณเก่ง

### ขั้น 6 — Demo forward test ≥ 1 เดือน
เทียบ log CSV (`MQL5/Files/SniperAI/decisions_*.csv`) กับผล backtest
ถ้า distribution ของ `meta_conf` ต่างกันชัดเจน = โมเดลเจอตลาดที่ไม่เคยเห็น

---

## 4. ค่า input ที่สำคัญ

| Input | ค่าเริ่มต้น | หมายเหตุ |
|---|---|---|
| `InpConfThreshold` | `0.60` | **ไม่ใช่ 0.35** — เลือกจาก threshold sweep |
| `InpMinPrimaryEdge` | `0.10` | กัน primary ที่ตอบ 50.5% แล้ว meta ปล่อยผ่าน |
| `InpStopMode` | `SNP_STOP_ATR` | SL/TP ตาม ATR — ดีกว่า `$` คงที่ (ดู REVIEW) |
| `InpTrailMode` | `SNP_TRAIL_ATR` | โหมด `SNP_TRAIL_MONEY` = พฤติกรรมเดิมของ V4 |
| `InpMaxSpreadPoints` | `350` | ทองสเปรดกว้างช่วงข่าว/rollover |
| `InpMaxEquityDDPct` | `5.0` | เบรกฉุกเฉิน ปิดทุกออเดอร์แล้วหยุดทั้งวัน |
| `InpAllowHeuristic` | `false` | ถ้าเปิด = รันด้วยสูตรจำลอง **ไม่ใช่โมเดลคุณ** ห้ามใช้เงินจริง |

ค่าใน Panel (conf / lot / TP$ / SL$ / trailing) แก้สดได้ระหว่างรัน —
มีผลทันทีกับไม้ถัดไป และเขียนลง log ทุกครั้ง

**Stop Bot ไม่ได้แปลว่าทิ้งออเดอร์** — หยุดคือ "ไม่เปิดไม้ใหม่"
ออเดอร์ที่เปิดอยู่ยังถูก trailing / break-even / ลิมิตรายวัน ดูแลต่อทุก tick

---

## 5. ถ้าจะเชื่อมต่อกับ Fusion Engine (Rust / C++ / CUDA)

คำตอบสั้นๆ: **ตอนนี้ยังไม่ต้อง** และเหตุผลอยู่ใน [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)
หัวข้อ "The Ultimate Bridge" — สรุปคือ decision loop ของคุณคือ **1 ครั้งต่อ 5 นาที**
ส่วน ONNX inference ของ tree ensemble ใช้เวลาระดับ **ไมโครวินาที**
การเอา CUDA มาเร่ง inference ที่ใช้เวลา 40 µs ทุก 300,000,000 µs ไม่ได้แก้ปัญหาอะไรเลย
คอขวดจริงอยู่ที่ latency ของโบรกเกอร์ ไม่ใช่ที่การคำนวณ

---

## 6. ข้อจำกัดที่ต้องรู้

- Tick volume ≠ real volume — ดู `docs/REVIEW.md` §3
- EA เทรดสัญลักษณ์เดียวต่อ 1 ชาร์ต แยก `InpMagic` ถ้าจะรันหลายตัว
- บัญชี **hedging** เท่านั้นถ้าตั้ง `InpMaxPositions > 1` (netting จะรวมไม้)
- ยังไม่มีตัวกรองข่าวอัตโนมัติ — ใช้ `InpBlackoutSpec` ตั้งช่วงเวลาเอง

## 7. คำเตือน

โค้ดชุดนี้คือ **โครงสร้าง** ที่ถูกต้อง ไม่ใช่ **กำไร** ที่รับประกัน
โครงสร้างดีแค่ไหนก็ไม่ช่วย ถ้าโมเดลข้างในไม่มี edge จริง
`tools/train_two_tier.py` จะบอกคุณตรงๆ ถ้ามันไม่มี — ให้ฟังมัน
