//+------------------------------------------------------------------+
//|                                                  SniperAI_V5.mq5 |
//|  Sniper AI V5 - native MQL5 rebuild of the "Sniper AI Control    |
//|  Panel V4" Python bot.                                           |
//|                                                                  |
//|  WHY THIS EXISTS                                                 |
//|  The Python version put the decision loop, the GUI thread and    |
//|  the trailing stop in one process outside the terminal. Any of   |
//|  the three could stall and leave a live position unmanaged.      |
//|  Here the whole pipeline - features, two-tier inference, risk,   |
//|  execution - runs inside MT5, every stop is server-side, and     |
//|  the same code runs in the Strategy Tester.                      |
//|                                                                  |
//|  READ docs/README.md BEFORE TOUCHING A LIVE ACCOUNT.             |
//+------------------------------------------------------------------+
#property copyright   "SniperAI V5"
#property link        "https://github.com/anon5547/MyBigBag"
#property version     "5.00"
#property description "Two-tier meta-labelling AI expert advisor (ONNX) with money-based risk control."

#include <SniperAI/Features.mqh>
#include <SniperAI/Inference.mqh>
#include <SniperAI/Broker.mqh>
#include <SniperAI/RiskManager.mqh>
#include <SniperAI/Filters.mqh>
#include <SniperAI/Panel.mqh>
#include <SniperAI/Logger.mqh>

//--- what to do when a signal arrives against an open position
enum ENUM_SNP_OPPOSITE
  {
   SNP_OPP_IGNORE  = 0,   // keep the position, skip the signal
   SNP_OPP_CLOSE   = 1,   // close, do not re-enter this bar
   SNP_OPP_REVERSE = 2    // close and enter the other way
  };

//==================== INPUTS ======================================
input group                "=== Model ==="
input string   InpPrimaryModel      = "SniperAI\\primary.onnx";   // Primary .onnx (MQL5\Files\...)
input string   InpMetaModel         = "SniperAI\\meta.onnx";      // Meta .onnx (MQL5\Files\...)
input int      InpUpClassIndex      = 1;                          // Index of the "UP" column in the output
input bool     InpSingleOutput      = false;                      // Model emits probabilities only (no label)
input bool     InpAllowHeuristic    = false;                      // DEMO ONLY: run without models

input group                "=== Signal ==="
input double   InpConfThreshold     = 0.60;   // Meta confidence threshold (0-1)
input double   InpMinPrimaryEdge    = 0.10;   // Minimum |2*P(up)-1| from the primary model
input bool     InpAllowBuy          = true;   // Allow BUY
input bool     InpAllowSell         = true;   // Allow SELL
input ENUM_TIMEFRAMES InpTimeframe  = PERIOD_M5;  // Decision timeframe
input int      InpHistoryBars       = 1200;   // Bars fed to the feature engine (>= 1000)
input ENUM_SNP_OPPOSITE InpOnOpposite = SNP_OPP_IGNORE;  // On an opposite signal

input group                "=== Position sizing ==="
input ENUM_SNP_LOT_MODE InpLotMode  = SNP_LOT_FIXED;  // Lot mode
input double   InpFixedLot          = 0.05;   // Fixed lot
input double   InpRiskPercent       = 0.5;    // Risk per trade (% of equity)
input double   InpMaxLot            = 1.00;   // Hard lot cap

input group                "=== Stops ==="
input ENUM_SNP_STOP_MODE InpStopMode = SNP_STOP_ATR;  // TP/SL mode
input double   InpTpMoney           = 100.0;  // TP Money ($)
input double   InpSlMoney           = 10.0;   // SL Money ($)
input double   InpTpPoints          = 4000;   // TP (points)
input double   InpSlPoints          = 2000;   // SL (points)
input double   InpTpAtrMult         = 2.0;    // TP = n x ATR(14)
input double   InpSlAtrMult         = 1.5;    // SL = n x ATR(14)

input group                "=== Trailing / break-even ==="
input ENUM_SNP_TRAIL_MODE InpTrailMode = SNP_TRAIL_ATR;  // Trailing mode
input double   InpTrailStartMoney   = 1.0;    // Trailing Start ($)
input double   InpTrailDistMoney    = 0.5;    // Trailing Distance ($)
input double   InpTrailStartAtr     = 1.0;    // Trailing start = n x ATR
input double   InpTrailDistAtr      = 1.0;    // Trailing distance = n x ATR
input bool     InpUseBreakeven      = true;   // Use break-even
input double   InpBeTriggerMoney    = 5.0;    // Break-even trigger ($)
input double   InpBeLockMoney       = 0.5;    // Lock in at break-even ($)

input group                "=== Guards ==="
input int      InpMaxPositions      = 1;      // Max concurrent positions
input int      InpMaxHoldBars       = 24;     // Force-close after N bars (0 = off)
input double   InpDailyLossLimit    = 0.0;    // Daily loss limit ($, 0 = off)
input double   InpDailyProfitTarget = 0.0;    // Daily profit target ($, 0 = off)
input double   InpMaxEquityDDPct    = 5.0;    // Max intraday equity drawdown (%, 0 = off)
input int      InpCooldownBars      = 2;      // Cooldown bars after a loss
input int      InpMaxSpreadPoints   = 350;    // Max spread (points, 0 = off)

input group                "=== Session / regime filters ==="
input bool     InpUseSessions       = true;                    // Use session filter
input string   InpSessionSpec       = "08:00-11:30,13:00-20:00"; // Sessions (SERVER time)
input string   InpBlackoutSpec      = "";                      // Blackout windows (SERVER time)
input bool     InpTradeMon = true;   // Monday
input bool     InpTradeTue = true;   // Tuesday
input bool     InpTradeWed = true;   // Wednesday
input bool     InpTradeThu = true;   // Thursday
input bool     InpTradeFri = true;   // Friday
input int      InpFridayCutoffMin   = 1200;   // Friday cut-off (minutes from midnight, 0 = off)
input int      InpRolloverGuardMin  = 10;     // Avoid +/- N minutes around 00:00 server
input double   InpMinAdx            = 18.0;   // Min ADX(14)  (0 = off)
input double   InpMaxAdx            = 0.0;    // Max ADX(14)  (0 = off)
input double   InpMaxAtrRatio       = 2.5;    // Max ATR(14)/ATR(50) (0 = off)
input double   InpMinAtrNorm        = 0.0;    // Min ATR(14)/close   (0 = off)

input group                "=== Execution ==="
input ulong    InpMagic             = 5150925; // Magic number
input int      InpDeviation         = 30;      // Max deviation (points)
input int      InpMaxRetries        = 3;       // Order retries
input string   InpComment           = "SniperAI_V5"; // Order comment

input group                "=== Interface / logging ==="
input bool     InpShowPanel         = true;   // Show the control panel
input bool     InpAutoStart         = false;  // Start trading immediately on attach
input int      InpPanelX            = 12;     // Panel X
input int      InpPanelY            = 24;     // Panel Y
input ENUM_SNP_LOGLEVEL InpLogLevel = SNP_LOG_INFO;  // Log level
input bool     InpWriteDecisionCsv  = true;   // Write the decision audit CSV

//==================== GLOBALS =====================================
CSnpLogger        g_log;
CSnpBroker        g_broker;
CSnpFeatureEngine g_features;
CSnpInference     g_ai;
CSnpRisk          g_risk;
CSnpFilters       g_filters;
CSnpPanel         g_panel;

bool     g_running        = false;
datetime g_last_bar       = 0;
int      g_bar_seconds    = 300;
int      g_same_side_run  = 0;
int      g_last_side      = 0;
double   g_ui_conf        = 0.60;
double   g_ui_lot         = 0.05;
double   g_ui_tp          = 100.0;
double   g_ui_sl          = 10.0;
double   g_ui_ts          = 1.0;
double   g_ui_td          = 0.5;
string   g_model_status   = "-";

//==================== HELPERS =====================================
SnpRiskConfig BuildRiskConfig()
  {
   SnpRiskConfig c;
   c.stop_mode        = InpStopMode;
   c.tp_money         = g_ui_tp;
   c.sl_money         = g_ui_sl;
   c.tp_points        = InpTpPoints;
   c.sl_points        = InpSlPoints;
   c.tp_atr_mult      = InpTpAtrMult;
   c.sl_atr_mult      = InpSlAtrMult;

   c.lot_mode         = InpLotMode;
   c.fixed_lot        = g_ui_lot;
   c.risk_percent     = InpRiskPercent;
   c.max_lot_cap      = InpMaxLot;

   c.trail_mode       = InpTrailMode;
   c.trail_start_money= g_ui_ts;
   c.trail_dist_money = g_ui_td;
   c.trail_start_atr  = InpTrailStartAtr;
   c.trail_dist_atr   = InpTrailDistAtr;

   c.use_breakeven    = InpUseBreakeven;
   c.be_trigger_money = InpBeTriggerMoney;
   c.be_lock_money    = InpBeLockMoney;

   c.max_positions    = InpMaxPositions;
   c.max_hold_bars    = InpMaxHoldBars;
   c.daily_loss_limit = InpDailyLossLimit;
   c.daily_profit_target = InpDailyProfitTarget;
   c.max_equity_dd_pct   = InpMaxEquityDDPct;
   c.cooldown_bars_after_loss = InpCooldownBars;

   c.magic            = InpMagic;
   c.deviation        = InpDeviation;
   c.max_spread_points= InpMaxSpreadPoints;
   return(c);
  }

SnpFilterConfig BuildFilterConfig()
  {
   SnpFilterConfig f;
   f.use_sessions       = InpUseSessions;
   f.session_spec       = InpSessionSpec;
   f.blackout_spec      = InpBlackoutSpec;
   f.trade_mon          = InpTradeMon;
   f.trade_tue          = InpTradeTue;
   f.trade_wed          = InpTradeWed;
   f.trade_thu          = InpTradeThu;
   f.trade_fri          = InpTradeFri;
   f.friday_cutoff_min  = InpFridayCutoffMin;
   f.rollover_guard_min = InpRolloverGuardMin;
   f.max_spread_points  = InpMaxSpreadPoints;
   f.min_adx            = InpMinAdx;
   f.max_adx            = InpMaxAdx;
   f.max_atr_ratio      = InpMaxAtrRatio;
   f.min_atr_norm       = InpMinAtrNorm;
   return(f);
  }

//+------------------------------------------------------------------+
void RefreshPanelStatus()
  {
   if(!g_panel.IsBuilt()) return;

   g_panel.SetStatus("model", g_model_status,
                     g_ai.IsFallback() ? clrOrange : (g_ai.IsReady() ? C'60,220,90' : clrOrangeRed));

   int    n_pos    = g_risk.CountOwnPositions();
   double realized = g_risk.RealizedToday();
   double floating = g_risk.FloatingOwn();
   double day      = realized + floating;

   g_panel.SetStatus("pos", StringFormat("%d / %d", n_pos, InpMaxPositions), clrWhite);
   g_panel.SetStatus("pnl", StringFormat("%.2f  (realised %.2f)", day, realized),
                     day >= 0.0 ? C'60,220,90' : C'240,90,90');

   int sp = g_broker.SpreadPoints();
   g_panel.SetStatus("spread", IntegerToString(sp),
                     (InpMaxSpreadPoints > 0 && sp > InpMaxSpreadPoints) ? C'240,90,90' : clrWhite);
  }

//+------------------------------------------------------------------+
void ApplyUiToEngine()
  {
   SnpRiskConfig c = BuildRiskConfig();
   g_risk.UpdateConfig(c);
  }

//==================== LIFECYCLE ===================================
int OnInit()
  {
   g_ui_conf = InpConfThreshold;
   g_ui_lot  = InpFixedLot;
   g_ui_tp   = InpTpMoney;
   g_ui_sl   = InpSlMoney;
   g_ui_ts   = InpTrailStartMoney;
   g_ui_td   = InpTrailDistMoney;

   string csv = "";
   if(InpWriteDecisionCsv && !MQLInfoInteger(MQL_OPTIMIZATION))
      csv = StringFormat("SniperAI\\decisions_%s_%s.csv", _Symbol, EnumToString(InpTimeframe));
   g_log.Configure(InpLogLevel, true, csv);

   g_log.Info("SYSTEM", "กำลังเริ่มต้น Sniper AI V5 ...");

   if(InpHistoryBars < SNP_WARMUP_BARS)
     {
      g_log.Error("SYSTEM", StringFormat("InpHistoryBars ต้อง >= %d", SNP_WARMUP_BARS));
      return(INIT_PARAMETERS_INCORRECT);
     }
   if(InpConfThreshold < 0.0 || InpConfThreshold > 1.0)
     {
      g_log.Error("SYSTEM", "InpConfThreshold ต้องอยู่ระหว่าง 0 ถึง 1");
      return(INIT_PARAMETERS_INCORRECT);
     }
   if(!InpAllowBuy && !InpAllowSell)
     {
      g_log.Error("SYSTEM", "ปิดทั้ง BUY และ SELL แล้วจะให้เทรดอะไร");
      return(INIT_PARAMETERS_INCORRECT);
     }

   if(!g_broker.Init(_Symbol, GetPointer(g_log)))
      return(INIT_FAILED);

   g_bar_seconds = PeriodSeconds(InpTimeframe);
   g_risk.Init(GetPointer(g_broker), GetPointer(g_log), BuildRiskConfig());
   g_filters.Configure(BuildFilterConfig());

   if(InpUseSessions && g_filters.SessionCount() == 0)
     {
      g_log.Error("SYSTEM", "รูปแบบเวลาเซสชันไม่ถูกต้อง: " + InpSessionSpec);
      return(INIT_PARAMETERS_INCORRECT);
     }

   //--- models -----------------------------------------------------
   if(g_ai.Load(InpPrimaryModel, InpMetaModel, InpUpClassIndex, InpSingleOutput))
     {
      g_model_status = "ONNX 2-tier พร้อม";
      g_log.Info("AI", "โหลดโมเดล ONNX ทั้งสองชั้นสำเร็จ");
     }
   else
     {
      g_log.Error("AI", g_ai.LastError());
      if(InpAllowHeuristic)
        {
         g_ai.EnableFallback(true);
         g_model_status = "HEURISTIC (ห้ามใช้เงินจริง)";
         g_log.Warn("AI", "*** โหมดจำลอง: ไม่ใช่โมเดลของคุณ ห้ามรันบัญชีจริงเด็ดขาด ***");
        }
      else
        {
         g_log.Error("AI", "ไม่มีโมเดล และไม่อนุญาตโหมดจำลอง -> หยุดทำงาน");
         return(INIT_FAILED);
        }
     }

   //--- panel ------------------------------------------------------
   if(InpShowPanel && !MQLInfoInteger(MQL_OPTIMIZATION))
     {
      g_panel.Create(ChartID(), InpPanelX, InpPanelY);
      g_panel.SyncInputs(g_ui_conf, g_ui_lot, g_ui_tp, g_ui_sl, g_ui_ts, g_ui_td);
      RefreshPanelStatus();
     }

   g_running = InpAutoStart;
   g_panel.SetRunning(g_running);
   g_log.Info("SYSTEM", g_running ? "บอทเริ่มทำงานอัตโนมัติแล้ว" : "พร้อมทำงาน - กด Start Bot เพื่อเริ่ม");

   //--- adopt any position we already own (terminal restart)
   int adopted = g_risk.CountOwnPositions();
   if(adopted > 0)
      g_log.Warn("SYSTEM", StringFormat("พบออเดอร์เดิมของ EA อยู่ %d ออเดอร์ - เข้าดูแลต่อ", adopted));

   //--- Prime the feature engine now, not on the first new bar. A position
   //--- adopted after a restart needs a valid ATR immediately or the
   //--- "no SL" repair in ManageOpenPositions cannot compute one.
     {
      MqlRates warm[];
      ArraySetAsSeries(warm, false);
      int got = CopyRates(_Symbol, InpTimeframe, 1, InpHistoryBars, warm);
      float wf[];
      if(got >= SNP_WARMUP_BARS && g_features.Build(warm, got, wf))
         g_log.Info("SYSTEM", StringFormat("ฟีเจอร์พร้อม | ATR(14)=%.5f | ADX=%.1f | RSI(14)=%.1f",
                                           g_features.LastATR(), g_features.LastADX(), g_features.LastRSI14()));
      else
         g_log.Warn("SYSTEM", StringFormat("ประวัติราคายังไม่พอ (%d/%d แท่ง) - เลื่อนกราฟย้อนหลังเพื่อโหลดข้อมูล",
                                           got, SNP_WARMUP_BARS));
     }

   g_last_bar = iTime(_Symbol, InpTimeframe, 0);
   EventSetTimer(2);
   return(INIT_SUCCEEDED);
  }

//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
   EventKillTimer();
   g_panel.Destroy();
   g_ai.Release();
   g_log.Info("SYSTEM", StringFormat("บอทหยุดทำงาน (reason=%d)", reason));
   g_log.Close();
  }

//+------------------------------------------------------------------+
void OnTimer()
  {
   if(!g_panel.IsBuilt()) return;
   RefreshPanelStatus();
   if(g_log.Dirty())
     {
      g_panel.RenderLog(g_log);
      g_log.ClearDirty();
     }
  }

//+------------------------------------------------------------------+
void OnChartEvent(const int id, const long &lparam, const double &dparam, const string &sparam)
  {
   if(!g_panel.IsBuilt()) return;

   if(id == CHARTEVENT_OBJECT_CLICK)
     {
      int action = g_panel.OnClick(sparam);
      if(action == 1 && !g_running)
        {
         g_running = true;
         g_panel.SetRunning(true);
         g_log.Info("SYSTEM", "บอท AI 'Sniper V5' เริ่มทำงานแล้ว...");
        }
      else if(action == -1 && g_running)
        {
         g_running = false;
         g_panel.SetRunning(false);
         g_log.Info("SYSTEM", "บอทหยุดทำงานเรียบร้อยแล้ว (ออเดอร์ที่เปิดอยู่ยังถูกดูแลต่อ)");
        }
      return;
     }

   if(id == CHARTEVENT_OBJECT_ENDEDIT)
     {
      ENUM_SNP_EDIT which = g_panel.OnEndEdit(sparam);
      if(which == SNP_EDIT_NONE) return;

      g_ui_conf = g_panel.Conf();
      g_ui_lot  = g_panel.Lot();
      g_ui_tp   = g_panel.TpMoney();
      g_ui_sl   = g_panel.SlMoney();
      g_ui_ts   = g_panel.TrailStart();
      g_ui_td   = g_panel.TrailDist();
      ApplyUiToEngine();

      g_log.Info("PANEL", StringFormat("ปรับค่า: conf=%.2f lot=%.2f TP=$%.2f SL=$%.2f trail=%.2f/%.2f",
                                       g_ui_conf, g_ui_lot, g_ui_tp, g_ui_sl, g_ui_ts, g_ui_td));

      if(g_ui_conf < 0.50)
         g_log.Warn("PANEL", "Threshold ต่ำกว่า 0.50 = ยิงแทบทุกสัญญาณ กรุณาอ่าน docs/REVIEW.md");
     }
  }

//+------------------------------------------------------------------+
//| Track closed deals so cooldown and the log see real outcomes.    |
//+------------------------------------------------------------------+
void OnTradeTransaction(const MqlTradeTransaction &trans,
                        const MqlTradeRequest &request,
                        const MqlTradeResult &result)
  {
   if(trans.type != TRADE_TRANSACTION_DEAL_ADD) return;
   if(!HistoryDealSelect(trans.deal)) return;
   if(HistoryDealGetInteger(trans.deal, DEAL_MAGIC) != (long)InpMagic) return;
   if(HistoryDealGetString(trans.deal, DEAL_SYMBOL) != _Symbol) return;

   ENUM_DEAL_ENTRY entry = (ENUM_DEAL_ENTRY)HistoryDealGetInteger(trans.deal, DEAL_ENTRY);
   if(entry != DEAL_ENTRY_OUT && entry != DEAL_ENTRY_OUT_BY) return;

   double pnl = HistoryDealGetDouble(trans.deal, DEAL_PROFIT)
              + HistoryDealGetDouble(trans.deal, DEAL_SWAP)
              + HistoryDealGetDouble(trans.deal, DEAL_COMMISSION);
   ENUM_DEAL_REASON reason = (ENUM_DEAL_REASON)HistoryDealGetInteger(trans.deal, DEAL_REASON);

   string why = "manual";
   if(reason == DEAL_REASON_SL)     why = "STOP LOSS";
   else if(reason == DEAL_REASON_TP) why = "TAKE PROFIT";
   else if(reason == DEAL_REASON_EXPERT) why = "EA";

   g_log.Info("CLOSED", StringFormat("ปิดออเดอร์ (%s) กำไร/ขาดทุน: $%.2f", why, pnl));

   if(pnl < 0.0)
      g_risk.MarkLoss(iTime(_Symbol, InpTimeframe, 0));
  }

//==================== CORE LOOP ===================================
void OnTick()
  {
   g_risk.NewDayCheck();

   //--- open positions are managed on EVERY tick, even when the bot
   //--- is "stopped". Stopping means "take no NEW trades", never
   //--- "abandon what is already open".
   double atr_now = g_features.LastATR();
   g_risk.ManageOpenPositions(atr_now, iTime(_Symbol, InpTimeframe, 0), g_bar_seconds);

   //--- everything below is per closed bar only
   datetime bar = iTime(_Symbol, InpTimeframe, 0);
   if(bar == 0 || bar == g_last_bar)
      return;
   g_last_bar = bar;

   if(!g_running)
      return;

   //=== 1. history ===============================================
   MqlRates rates[];
   ArraySetAsSeries(rates, false);
   int copied = CopyRates(_Symbol, InpTimeframe, 1, InpHistoryBars, rates);
   if(copied < SNP_WARMUP_BARS)
     {
      g_log.Warn("DATA", StringFormat("ข้อมูลไม่พอ (%d/%d แท่ง) - รอโหลดกราฟ", copied, SNP_WARMUP_BARS));
      return;
     }
   datetime signal_bar = rates[copied - 1].time;

   //=== 2. features ==============================================
   float f[];
   if(!g_features.Build(rates, copied, f))
     {
      g_log.Warn("FEATURES", "สร้างฟีเจอร์ไม่สำเร็จ: " + g_features.Error());
      return;
     }

   g_log.Debug("M5 Bar", "วิเคราะห์แท่งเทียน: " + TimeToString(signal_bar, TIME_DATE | TIME_MINUTES));

   //=== 3. inference =============================================
   SnpPrediction p;
   if(!g_ai.Predict(f, p))
     {
      g_log.Error("AI", "อนุมานไม่สำเร็จ: " + p.reason);
      return;
     }

   g_log.Info("AI Stats", StringFormat("BUY: %.2f%% | SELL: %.2f%% | Meta Confidence: %.2f",
                                       p.p_up * 100.0, p.p_down * 100.0, p.meta_conf));
   if(g_panel.IsBuilt())
     {
      g_panel.SetStatus("signal", (p.side == SNP_SIDE_BUY) ? "BUY" : "SELL",
                        (p.side == SNP_SIDE_BUY) ? C'60,220,90' : C'240,120,120');
      g_panel.SetStatus("prob", StringFormat("%.1f%% / %.1f%%", p.p_up * 100.0, p.p_down * 100.0));
      g_panel.SetStatus("meta", StringFormat("%.3f  (>= %.2f)", p.meta_conf, g_ui_conf),
                        p.meta_conf >= g_ui_conf ? C'60,220,90' : C'200,200,120');
     }

   //--- one-sided-model canary. In the V4 logs EVERY signal was SELL.
   //--- That is not an edge, that is a broken label or a leaked trend.
   int side_i = (int)p.side;
   if(side_i == g_last_side) g_same_side_run++;
   else                      { g_same_side_run = 1; g_last_side = side_i; }
   if(g_same_side_run == 25)
      g_log.Warn("AI", "เตือน: สัญญาณออกทางเดียวกัน 25 ครั้งติด - ตรวจสอบ label/มาตราส่วนฟีเจอร์ด่วน");

   //=== 4. gates =================================================
   string gate = "";
   string halt_reason = "";

   if(g_risk.TradingHalted(halt_reason))
      gate = halt_reason;
   else if(!g_broker.MarketOpen())
      gate = "ตลาดปิด / ไม่มีราคา";
   else if(!g_filters.TimeAllowed(TimeCurrent()))
      gate = g_filters.LastReason();
   else if(!g_filters.SpreadAllowed(g_broker.SpreadPoints()))
      gate = g_filters.LastReason();
   else if(!g_filters.RegimeAllowed(f[F_ADX_14], f[F_ATR_RATIO], f[F_ATR14_NORM]))
      gate = g_filters.LastReason();
   else if(g_risk.InCooldown(bar, g_bar_seconds))
      gate = "อยู่ในช่วงพักหลังขาดทุน";
   else if(p.edge < InpMinPrimaryEdge)
      gate = StringFormat("Primary ไม่ชี้ชัด (edge %.3f < %.3f)", p.edge, InpMinPrimaryEdge);
   else if(p.meta_conf < g_ui_conf)
      gate = StringFormat("ความมั่นใจไม่ถึงเกณฑ์ (%.2f < %.2f)", p.meta_conf, g_ui_conf);
   else if(p.side == SNP_SIDE_BUY  && !InpAllowBuy)
      gate = "ปิดฝั่ง BUY ไว้";
   else if(p.side == SNP_SIDE_SELL && !InpAllowSell)
      gate = "ปิดฝั่ง SELL ไว้";

   if(g_panel.IsBuilt())
      g_panel.SetStatus("gate", gate == "" ? "ผ่าน" : gate, gate == "" ? C'60,220,90' : C'200,200,120');

   if(gate != "")
     {
      g_log.Info("Filter Refused", gate);
      g_log.Decision(signal_bar, _Symbol, p.p_up, p.meta_conf, "REFUSED", gate, f);
      return;
     }

   //=== 5. opposite-position policy ==============================
   ENUM_ORDER_TYPE otype = (p.side == SNP_SIDE_BUY) ? ORDER_TYPE_BUY : ORDER_TYPE_SELL;
   bool blocked_by_opposite = false;

   for(int i = PositionsTotal() - 1; i >= 0; i--)
     {
      ulong t = PositionGetTicket(i);
      if(t == 0) continue;
      if(PositionGetInteger(POSITION_MAGIC) != (long)InpMagic) continue;
      if(PositionGetString(POSITION_SYMBOL) != _Symbol) continue;

      ENUM_POSITION_TYPE pt = (ENUM_POSITION_TYPE)PositionGetInteger(POSITION_TYPE);
      bool opposite = (pt == POSITION_TYPE_BUY  && otype == ORDER_TYPE_SELL) ||
                      (pt == POSITION_TYPE_SELL && otype == ORDER_TYPE_BUY);
      if(!opposite) continue;

      if(InpOnOpposite == SNP_OPP_IGNORE)
        {
         blocked_by_opposite = true;
         break;
        }
      g_log.Info("SIGNAL", StringFormat("สัญญาณสวนทาง -> ปิดออเดอร์ #%I64u", t));
      g_broker.ClosePosition(t, InpDeviation, InpMaxRetries);
      if(InpOnOpposite == SNP_OPP_CLOSE)
         blocked_by_opposite = true;
     }

   if(blocked_by_opposite)
     {
      g_log.Info("Filter Refused", "มีออเดอร์สวนทางอยู่แล้ว");
      g_log.Decision(signal_bar, _Symbol, p.p_up, p.meta_conf, "REFUSED", "opposite position", f);
      return;
     }

   if(g_risk.CountOwnPositions() >= InpMaxPositions)
     {
      g_log.Info("Filter Refused", StringFormat("ถือครบลิมิตแล้ว (%d)", InpMaxPositions));
      g_log.Decision(signal_bar, _Symbol, p.p_up, p.meta_conf, "REFUSED", "max positions", f);
      return;
     }

   //=== 6. size and stops ========================================
   double atr = g_features.LastATR();
   double lot = g_risk.ResolveLot(atr);
   if(lot <= 0.0)
     {
      g_log.Error("RISK", "คำนวณ lot ไม่ได้");
      return;
     }

   double entry = (otype == ORDER_TYPE_BUY) ? g_broker.Ask() : g_broker.Bid();
   double sl = 0.0, tp = 0.0;
   if(!g_risk.ResolveStops(otype, entry, lot, atr, sl, tp))
     {
      g_log.Error("RISK", "คำนวณ SL/TP ไม่ได้ - ยกเลิกการยิงออเดอร์ (ห้ามเปิดออเดอร์ไร้ SL)");
      return;
     }

   //--- refuse a stop the broker would have to widen beyond reason
   double want_sl = sl;
   g_broker.ClampStops(otype, entry, sl, tp);
   double widened = MathAbs(sl - want_sl);
   if(widened > MathAbs(entry - want_sl) * 0.5)
     {
      g_log.Warn("RISK", StringFormat("โบรกเกอร์บังคับขยาย SL เกิน 50%% (stops level %d จุด) - ข้ามไม้นี้",
                                      g_broker.StopsLevel()));
      g_log.Decision(signal_bar, _Symbol, p.p_up, p.meta_conf, "REFUSED", "stops level too wide", f);
      return;
     }

   double risk_cash = g_broker.PriceDistanceToMoney(MathAbs(entry - sl), lot);

   //=== 7. fire ==================================================
   g_log.Info("Signal Triggered", StringFormat("ผ่านเกณฑ์! เตรียมยิงออเดอร์ %s",
                                               (otype == ORDER_TYPE_BUY) ? "BUY" : "SELL"));

   SnpOrderResult res = g_broker.Open(otype, lot, sl, tp, InpMagic, InpComment,
                                      InpDeviation, InpMaxRetries);
   if(res.ok)
     {
      g_log.Info("MT5", StringFormat("เปิด %s สำเร็จ | Ticket: %I64u | ราคา: %s | Lot: %s | SL: %s | ความเสี่ยง: $%.2f",
                                     (otype == ORDER_TYPE_BUY) ? "BUY" : "SELL",
                                     res.ticket,
                                     DoubleToString(res.price, g_broker.Digits()),
                                     DoubleToString(res.volume, 2),
                                     DoubleToString(sl, g_broker.Digits()),
                                     risk_cash));
      g_log.Decision(signal_bar, _Symbol, p.p_up, p.meta_conf, "OPEN",
                     StringFormat("%s lot=%.2f sl=%.5f tp=%.5f",
                                  (otype == ORDER_TYPE_BUY) ? "BUY" : "SELL", res.volume, sl, tp), f);
     }
   else
     {
      g_log.Error("MT5", "เปิดออเดอร์ไม่สำเร็จ: " + res.message);
      g_log.Decision(signal_bar, _Symbol, p.p_up, p.meta_conf, "FAILED", res.message, f);
     }
  }

//+------------------------------------------------------------------+
//| Optimisation criterion.                                          |
//|                                                                  |
//| Net profit alone rewards one lucky trade and 500 lot martingales.|
//| This one is expectancy per trade, penalised by drawdown and by   |
//| a sample too small to mean anything.                             |
//+------------------------------------------------------------------+
double OnTester()
  {
   double trades = TesterStatistics(STAT_TRADES);
   if(trades < 60.0) return(0.0);

   double profit = TesterStatistics(STAT_PROFIT);
   double dd     = TesterStatistics(STAT_EQUITY_DDREL_PERCENT);
   double expect = profit / trades;

   if(dd < 0.5) dd = 0.5;
   double score = expect * MathSqrt(trades) / dd;
   return(MathIsValidNumber(score) ? score : 0.0);
  }
//+------------------------------------------------------------------+
