//+------------------------------------------------------------------+
//|                                                  RiskManager.mqh |
//|  SniperAI V5 - position sizing, stops, trailing, daily guards.   |
//|                                                                  |
//|  Design rule that the Python version got wrong:                  |
//|  EVERY stop is placed SERVER-SIDE as a real SL/TP on the         |
//|  position. The client-side money checks are a BACKUP, not the    |
//|  mechanism. If the terminal dies mid-trade, the broker still     |
//|  holds your stop. A trailing stop that only exists inside a      |
//|  Python while-loop is not a stop, it is a wish.                  |
//+------------------------------------------------------------------+
#property copyright "SniperAI V5"
#ifndef __SNIPER_RISK_MQH__
#define __SNIPER_RISK_MQH__

#include <SniperAI/Broker.mqh>
#include <SniperAI/Logger.mqh>

enum ENUM_SNP_STOP_MODE
  {
   SNP_STOP_MONEY  = 0,   // TP/SL expressed in account currency
   SNP_STOP_POINTS = 1,   // TP/SL expressed in points
   SNP_STOP_ATR    = 2    // TP/SL as a multiple of ATR(14)  <-- recommended
  };

enum ENUM_SNP_LOT_MODE
  {
   SNP_LOT_FIXED    = 0,
   SNP_LOT_RISK_PCT = 1   // size so that a full stop-out costs risk% of equity
  };

enum ENUM_SNP_TRAIL_MODE
  {
   SNP_TRAIL_OFF   = 0,
   SNP_TRAIL_MONEY = 1,   // the original panel's behaviour
   SNP_TRAIL_ATR   = 2
  };

//+------------------------------------------------------------------+
struct SnpRiskConfig
  {
   ENUM_SNP_STOP_MODE  stop_mode;
   double              tp_money,  sl_money;
   double              tp_points, sl_points;
   double              tp_atr_mult, sl_atr_mult;

   ENUM_SNP_LOT_MODE   lot_mode;
   double              fixed_lot;
   double              risk_percent;
   double              max_lot_cap;

   ENUM_SNP_TRAIL_MODE trail_mode;
   double              trail_start_money, trail_dist_money;
   double              trail_start_atr,   trail_dist_atr;

   bool                use_breakeven;
   double              be_trigger_money;
   double              be_lock_money;

   int                 max_positions;
   int                 max_hold_bars;
   double              daily_loss_limit;      // 0 = off
   double              daily_profit_target;   // 0 = off
   double              max_equity_dd_pct;     // 0 = off
   int                 cooldown_bars_after_loss;

   ulong               magic;
   int                 deviation;
   int                 max_spread_points;
  };

//+------------------------------------------------------------------+
struct SnpPosState
  {
   ulong             ticket;
   datetime          open_time;
   double            peak_profit;
   bool              breakeven_done;
   bool              seen;
  };

//+------------------------------------------------------------------+
class CSnpRisk
  {
private:
   SnpRiskConfig     m_cfg;
   CSnpBroker       *m_broker;
   CSnpLogger       *m_log;
   SnpPosState       m_state[];
   double            m_day_start_equity;
   datetime          m_day_start;
   bool              m_halted_today;
   string            m_halt_reason;
   datetime          m_last_loss_bar;

   int               FindState(const ulong ticket);
   int               TouchState(const ulong ticket, const datetime open_time);
   void              PruneStates();
   double            PositionProfitNet(const ulong ticket) const;

public:
                     CSnpRisk();
   void              Init(CSnpBroker *broker, CSnpLogger *logger, const SnpRiskConfig &cfg);
   void              UpdateConfig(const SnpRiskConfig &cfg) { m_cfg = cfg; }
   SnpRiskConfig     Config() const { return(m_cfg); }

   //--- sizing & stops ---------------------------------------------
   double            ResolveLot(const double atr) const;
   bool              ResolveStops(const ENUM_ORDER_TYPE type, const double entry,
                                  const double lot, const double atr,
                                  double &sl, double &tp) const;

   //--- gates -------------------------------------------------------
   int               CountOwnPositions() const;
   double            RealizedToday() const;
   double            FloatingOwn() const;
   bool              TradingHalted(string &reason);
   void              NewDayCheck();
   bool              InCooldown(const datetime current_bar, const int bar_seconds) const;
   void              MarkLoss(const datetime bar_time) { m_last_loss_bar = bar_time; }

   //--- per-tick management ----------------------------------------
   void              ManageOpenPositions(const double atr, const datetime current_bar,
                                         const int bar_seconds);
   void              CloseAllOwn(const string why);

   double            DayStartEquity() const { return(m_day_start_equity); }
   bool              IsHalted()       const { return(m_halted_today);     }
   string            HaltReason()     const { return(m_halt_reason);      }
  };

//+------------------------------------------------------------------+
CSnpRisk::CSnpRisk()
  {
   m_broker           = NULL;
   m_log              = NULL;
   m_day_start_equity = 0.0;
   m_day_start        = 0;
   m_halted_today     = false;
   m_halt_reason      = "";
   m_last_loss_bar    = 0;
   ArrayResize(m_state, 0);
  }

//+------------------------------------------------------------------+
void CSnpRisk::Init(CSnpBroker *broker, CSnpLogger *logger, const SnpRiskConfig &cfg)
  {
   m_broker = broker;
   m_log    = logger;
   m_cfg    = cfg;
   m_day_start_equity = AccountInfoDouble(ACCOUNT_EQUITY);
   MqlDateTime dt;
   TimeToStruct(TimeCurrent(), dt);
   dt.hour = 0; dt.min = 0; dt.sec = 0;
   m_day_start   = StructToTime(dt);
   m_halted_today = false;
   m_halt_reason  = "";
  }

//+------------------------------------------------------------------+
//| Roll the daily counters over at broker midnight.                 |
//+------------------------------------------------------------------+
void CSnpRisk::NewDayCheck()
  {
   MqlDateTime dt;
   TimeToStruct(TimeCurrent(), dt);
   dt.hour = 0; dt.min = 0; dt.sec = 0;
   datetime today = StructToTime(dt);
   if(today != m_day_start)
     {
      m_day_start        = today;
      m_day_start_equity = AccountInfoDouble(ACCOUNT_EQUITY);
      m_halted_today     = false;
      m_halt_reason      = "";
      if(m_log != NULL)
         m_log.Info("RISK", StringFormat("วันใหม่ | รีเซ็ตลิมิตรายวัน | Equity ตั้งต้น: %.2f", m_day_start_equity));
     }
  }

//+------------------------------------------------------------------+
double CSnpRisk::ResolveLot(const double atr) const
  {
   if(m_broker == NULL) return(0.0);

   double lot = m_cfg.fixed_lot;

   if(m_cfg.lot_mode == SNP_LOT_RISK_PCT)
     {
      double equity    = AccountInfoDouble(ACCOUNT_EQUITY);
      double risk_cash = equity * m_cfg.risk_percent / 100.0;

      //--- how far, in raw price, is a full stop-out?
      double sl_dist = 0.0;
      if(m_cfg.stop_mode == SNP_STOP_ATR)
         sl_dist = m_cfg.sl_atr_mult * atr;
      else if(m_cfg.stop_mode == SNP_STOP_POINTS)
         sl_dist = m_cfg.sl_points * m_broker.Point();
      else
        {
         //--- Money mode: the loss is ALREADY pinned at sl_money whatever
         //--- the lot is, so "risk %" can only mean one thing here - scale
         //--- the fixed lot down until sl_money fits inside the budget.
         //---
         //---     lot = fixed_lot * min(sl_money, risk_cash) / sl_money
         //---
         if(m_cfg.sl_money > SNP_EPS && m_cfg.fixed_lot > SNP_EPS)
            lot = m_cfg.fixed_lot * MathMin(m_cfg.sl_money, risk_cash) / m_cfg.sl_money;
         else
            lot = m_cfg.fixed_lot;
         sl_dist = 0.0;
        }

      if(sl_dist > SNP_EPS)
        {
         double per_unit_1lot = m_broker.MoneyPerPriceUnit(1.0);
         if(per_unit_1lot > SNP_EPS)
            lot = risk_cash / (sl_dist * per_unit_1lot);
        }
     }

   if(m_cfg.max_lot_cap > 0.0)
      lot = MathMin(lot, m_cfg.max_lot_cap);
   return(m_broker.NormalizeLot(lot));
  }

//+------------------------------------------------------------------+
bool CSnpRisk::ResolveStops(const ENUM_ORDER_TYPE type, const double entry,
                            const double lot, const double atr,
                            double &sl, double &tp) const
  {
   sl = 0.0; tp = 0.0;
   if(m_broker == NULL || lot <= 0.0) return(false);

   double sl_dist = 0.0, tp_dist = 0.0;

   switch(m_cfg.stop_mode)
     {
      case SNP_STOP_MONEY:
         sl_dist = m_broker.MoneyToPriceDistance(m_cfg.sl_money, lot);
         tp_dist = m_broker.MoneyToPriceDistance(m_cfg.tp_money, lot);
         break;
      case SNP_STOP_POINTS:
         sl_dist = m_cfg.sl_points * m_broker.Point();
         tp_dist = m_cfg.tp_points * m_broker.Point();
         break;
      case SNP_STOP_ATR:
         if(atr <= SNP_EPS) return(false);
         sl_dist = m_cfg.sl_atr_mult * atr;
         tp_dist = m_cfg.tp_atr_mult * atr;
         break;
     }

   if(sl_dist <= SNP_EPS) return(false);      // a position without a stop never leaves this function

   if(type == ORDER_TYPE_BUY)
     {
      sl = entry - sl_dist;
      tp = (tp_dist > SNP_EPS) ? entry + tp_dist : 0.0;
     }
   else
     {
      sl = entry + sl_dist;
      tp = (tp_dist > SNP_EPS) ? entry - tp_dist : 0.0;
     }
   return(true);
  }

//+------------------------------------------------------------------+
int CSnpRisk::CountOwnPositions() const
  {
   int n = 0;
   for(int i = PositionsTotal() - 1; i >= 0; i--)
     {
      ulong t = PositionGetTicket(i);
      if(t == 0) continue;
      if(PositionGetInteger(POSITION_MAGIC) != (long)m_cfg.magic) continue;
      if(PositionGetString(POSITION_SYMBOL) != m_broker.Symbol())  continue;
      n++;
     }
   return(n);
  }

//+------------------------------------------------------------------+
double CSnpRisk::FloatingOwn() const
  {
   double p = 0.0;
   for(int i = PositionsTotal() - 1; i >= 0; i--)
     {
      ulong t = PositionGetTicket(i);
      if(t == 0) continue;
      if(PositionGetInteger(POSITION_MAGIC) != (long)m_cfg.magic) continue;
      if(PositionGetString(POSITION_SYMBOL) != m_broker.Symbol())  continue;
      p += PositionGetDouble(POSITION_PROFIT) + PositionGetDouble(POSITION_SWAP);
     }
   return(p);
  }

//+------------------------------------------------------------------+
//| Realised P/L booked today by THIS ea on THIS symbol.             |
//| Read from deal history, so it survives a terminal restart.       |
//+------------------------------------------------------------------+
double CSnpRisk::RealizedToday() const
  {
   if(!HistorySelect(m_day_start, TimeCurrent() + 60))
      return(0.0);

   double total = 0.0;
   int deals = HistoryDealsTotal();
   for(int i = 0; i < deals; i++)
     {
      ulong d = HistoryDealGetTicket(i);
      if(d == 0) continue;
      if(HistoryDealGetInteger(d, DEAL_MAGIC) != (long)m_cfg.magic) continue;
      if(HistoryDealGetString(d, DEAL_SYMBOL) != m_broker.Symbol()) continue;
      ENUM_DEAL_ENTRY entry = (ENUM_DEAL_ENTRY)HistoryDealGetInteger(d, DEAL_ENTRY);
      if(entry != DEAL_ENTRY_OUT && entry != DEAL_ENTRY_OUT_BY && entry != DEAL_ENTRY_INOUT)
         continue;
      total += HistoryDealGetDouble(d, DEAL_PROFIT)
             + HistoryDealGetDouble(d, DEAL_SWAP)
             + HistoryDealGetDouble(d, DEAL_COMMISSION);
     }
   return(total);
  }

//+------------------------------------------------------------------+
bool CSnpRisk::TradingHalted(string &reason)
  {
   reason = "";
   if(m_halted_today)
     {
      reason = m_halt_reason;
      return(true);
     }

   double realized = RealizedToday();

   if(m_cfg.daily_loss_limit > 0.0 && realized <= -m_cfg.daily_loss_limit)
     {
      m_halted_today = true;
      m_halt_reason  = StringFormat("ชนลิมิตขาดทุนรายวัน (%.2f / -%.2f)", realized, m_cfg.daily_loss_limit);
      reason = m_halt_reason;
      if(m_log != NULL) m_log.Warn("RISK", "หยุดเทรดวันนี้: " + reason);
      return(true);
     }

   if(m_cfg.daily_profit_target > 0.0 && realized >= m_cfg.daily_profit_target)
     {
      m_halted_today = true;
      m_halt_reason  = StringFormat("ถึงเป้ากำไรรายวัน (%.2f / %.2f)", realized, m_cfg.daily_profit_target);
      reason = m_halt_reason;
      if(m_log != NULL) m_log.Info("RISK", "หยุดเทรดวันนี้: " + reason);
      return(true);
     }

   if(m_cfg.max_equity_dd_pct > 0.0 && m_day_start_equity > 0.0)
     {
      double eq = AccountInfoDouble(ACCOUNT_EQUITY);
      double dd = 100.0 * (m_day_start_equity - eq) / m_day_start_equity;
      if(dd >= m_cfg.max_equity_dd_pct)
        {
         m_halted_today = true;
         m_halt_reason  = StringFormat("Equity drawdown %.2f%% เกินลิมิต %.2f%%", dd, m_cfg.max_equity_dd_pct);
         reason = m_halt_reason;
         if(m_log != NULL) m_log.Error("RISK", "หยุดเทรดวันนี้: " + reason);
         CloseAllOwn("equity drawdown guard");
         return(true);
        }
     }
   return(false);
  }

//+------------------------------------------------------------------+
bool CSnpRisk::InCooldown(const datetime current_bar, const int bar_seconds) const
  {
   if(m_cfg.cooldown_bars_after_loss <= 0 || m_last_loss_bar == 0) return(false);
   long elapsed = (long)(current_bar - m_last_loss_bar);
   return(elapsed < (long)m_cfg.cooldown_bars_after_loss * bar_seconds);
  }

//+------------------------------------------------------------------+
int CSnpRisk::FindState(const ulong ticket)
  {
   for(int i = 0; i < ArraySize(m_state); i++)
      if(m_state[i].ticket == ticket) return(i);
   return(-1);
  }

int CSnpRisk::TouchState(const ulong ticket, const datetime open_time)
  {
   int idx = FindState(ticket);
   if(idx >= 0)
     {
      m_state[idx].seen = true;
      return(idx);
     }
   int n = ArraySize(m_state);
   ArrayResize(m_state, n + 1);
   m_state[n].ticket         = ticket;
   m_state[n].open_time      = open_time;
   m_state[n].peak_profit    = 0.0;
   m_state[n].breakeven_done = false;
   m_state[n].seen           = true;
   return(n);
  }

void CSnpRisk::PruneStates()
  {
   int w = 0;
   for(int i = 0; i < ArraySize(m_state); i++)
     {
      if(!m_state[i].seen) continue;
      if(w != i) m_state[w] = m_state[i];
      m_state[w].seen = false;
      w++;
     }
   ArrayResize(m_state, w);
  }

//+------------------------------------------------------------------+
double CSnpRisk::PositionProfitNet(const ulong ticket) const
  {
   if(!PositionSelectByTicket(ticket)) return(0.0);
   return(PositionGetDouble(POSITION_PROFIT) + PositionGetDouble(POSITION_SWAP));
  }

//+------------------------------------------------------------------+
//| Called on every tick. Trailing, break-even, max-hold, and the    |
//| client-side money net that catches anything the server missed.   |
//+------------------------------------------------------------------+
void CSnpRisk::ManageOpenPositions(const double atr, const datetime current_bar, const int bar_seconds)
  {
   if(m_broker == NULL) return;

   for(int i = PositionsTotal() - 1; i >= 0; i--)
     {
      ulong ticket = PositionGetTicket(i);
      if(ticket == 0) continue;
      if(PositionGetInteger(POSITION_MAGIC) != (long)m_cfg.magic) continue;
      if(PositionGetString(POSITION_SYMBOL) != m_broker.Symbol())  continue;

      ENUM_POSITION_TYPE ptype = (ENUM_POSITION_TYPE)PositionGetInteger(POSITION_TYPE);
      double   entry   = PositionGetDouble(POSITION_PRICE_OPEN);
      double   cur     = PositionGetDouble(POSITION_PRICE_CURRENT);
      double   vol     = PositionGetDouble(POSITION_VOLUME);
      double   cur_sl  = PositionGetDouble(POSITION_SL);
      double   cur_tp  = PositionGetDouble(POSITION_TP);
      double   profit  = PositionGetDouble(POSITION_PROFIT) + PositionGetDouble(POSITION_SWAP);
      datetime otime   = (datetime)PositionGetInteger(POSITION_TIME);

      int si = TouchState(ticket, otime);
      if(profit > m_state[si].peak_profit)
         m_state[si].peak_profit = profit;

      //=== 0. a position with no stop at all is an emergency =========
      if(cur_sl <= 0.0)
        {
         double sl2 = 0.0, tp2 = 0.0;
         if(ResolveStops((ptype == POSITION_TYPE_BUY) ? ORDER_TYPE_BUY : ORDER_TYPE_SELL,
                         entry, vol, atr, sl2, tp2))
           {
            if(cur_tp > 0.0) tp2 = cur_tp;
            if(m_broker.ModifyStops(ticket, sl2, tp2))
              {
               cur_sl = m_broker.NormalizePrice(sl2);   // do not let the
               cur_tp = (tp2 > 0.0) ? m_broker.NormalizePrice(tp2) : cur_tp;
               if(m_log != NULL)                        // blocks below read
                  m_log.Warn("RISK", StringFormat("ออเดอร์ #%I64u ไม่มี SL - ใส่ให้แล้ว", ticket));
              }                                         // a stale cur_sl
           }
        }

      //=== 1. client-side money net (BACKUP to the server stop) ======
      if(m_cfg.stop_mode == SNP_STOP_MONEY)
        {
         if(m_cfg.sl_money > 0.0 && profit <= -m_cfg.sl_money * 1.15)
           {
            if(m_log != NULL)
               m_log.Warn("MONEY", StringFormat("ตาข่ายกันพลาด: ปิด #%I64u ที่ %.2f (SL $%.2f)", ticket, profit, m_cfg.sl_money));
            if(m_broker.ClosePosition(ticket, m_cfg.deviation, 3))
              { MarkLoss(current_bar); continue; }
           }
         if(m_cfg.tp_money > 0.0 && profit >= m_cfg.tp_money)
           {
            if(m_log != NULL)
               m_log.Info("MONEY", StringFormat("ปิดออเดอร์ #%I64u เป้าหมายเงิน: $%.2f", ticket, profit));
            if(m_broker.ClosePosition(ticket, m_cfg.deviation, 3))
               continue;
           }
        }

      //=== 2. break-even =============================================
      if(m_cfg.use_breakeven && !m_state[si].breakeven_done &&
         m_cfg.be_trigger_money > 0.0 && profit >= m_cfg.be_trigger_money)
        {
         double lock_dist = m_broker.MoneyToPriceDistance(MathMax(0.0, m_cfg.be_lock_money), vol);
         double be_sl = (ptype == POSITION_TYPE_BUY) ? entry + lock_dist : entry - lock_dist;
         bool improves = (ptype == POSITION_TYPE_BUY) ? (cur_sl <= 0.0 || be_sl > cur_sl)
                                                      : (cur_sl <= 0.0 || be_sl < cur_sl);
         if(improves && m_broker.ModifyStops(ticket, be_sl, cur_tp))
           {
            //--- the trailing block below compares against cur_sl; if we
            //--- leave it stale it will happily move the stop backwards.
            cur_sl = m_broker.NormalizePrice(be_sl);
            m_state[si].breakeven_done = true;
            if(m_log != NULL)
               m_log.Info("BREAKEVEN", StringFormat("ย้าย SL ออเดอร์ #%I64u มาที่ทุน + $%.2f", ticket, m_cfg.be_lock_money));
           }
        }

      //=== 3. trailing stop =========================================
      if(m_cfg.trail_mode != SNP_TRAIL_OFF)
        {
         double new_sl = 0.0;
         bool   want   = false;

         if(m_cfg.trail_mode == SNP_TRAIL_MONEY)
           {
            //--- exactly the panel's semantics: once floating profit
            //--- reaches trail_start, keep the stop trail_dist behind
            //--- the profit HIGH-WATER MARK, expressed in dollars.
            if(m_cfg.trail_start_money > 0.0 && m_state[si].peak_profit >= m_cfg.trail_start_money)
              {
               double lock_money = m_state[si].peak_profit - m_cfg.trail_dist_money;
               if(lock_money > 0.0)
                 {
                  double dist = m_broker.MoneyToPriceDistance(lock_money, vol);
                  new_sl = (ptype == POSITION_TYPE_BUY) ? entry + dist : entry - dist;
                  want   = true;
                 }
              }
           }
         else if(m_cfg.trail_mode == SNP_TRAIL_ATR && atr > SNP_EPS)
           {
            double moved = (ptype == POSITION_TYPE_BUY) ? (cur - entry) : (entry - cur);
            if(moved >= m_cfg.trail_start_atr * atr)
              {
               double dist = m_cfg.trail_dist_atr * atr;
               new_sl = (ptype == POSITION_TYPE_BUY) ? cur - dist : cur + dist;
               want   = true;
              }
           }

         if(want)
           {
            bool improves = (ptype == POSITION_TYPE_BUY) ? (cur_sl <= 0.0 || new_sl > cur_sl + m_broker.Point() * 0.5)
                                                         : (cur_sl <= 0.0 || new_sl < cur_sl - m_broker.Point() * 0.5);
            if(improves && m_broker.ModifyStops(ticket, new_sl, cur_tp))
              {
               double locked = m_broker.PriceDistanceToMoney(
                                  (ptype == POSITION_TYPE_BUY) ? (new_sl - entry) : (entry - new_sl), vol);
               if(m_log != NULL)
                  m_log.Info("TRAILING", StringFormat("ล็อกกำไรออเดอร์ %I64u ที่: $%.2f", ticket, locked));
              }
           }
        }

      //=== 4. max hold time =========================================
      if(m_cfg.max_hold_bars > 0 && bar_seconds > 0)
        {
         long held = (long)(TimeCurrent() - otime);
         if(held >= (long)m_cfg.max_hold_bars * bar_seconds)
           {
            if(m_log != NULL)
               m_log.Info("RISK", StringFormat("ครบเวลาถือ %d แท่ง | ปิด #%I64u ที่ $%.2f",
                                               m_cfg.max_hold_bars, ticket, profit));
            if(m_broker.ClosePosition(ticket, m_cfg.deviation, 3))
              {
               if(profit < 0.0) MarkLoss(current_bar);
               continue;
              }
           }
        }
     }

   PruneStates();
  }

//+------------------------------------------------------------------+
void CSnpRisk::CloseAllOwn(const string why)
  {
   if(m_broker == NULL) return;
   for(int i = PositionsTotal() - 1; i >= 0; i--)
     {
      ulong t = PositionGetTicket(i);
      if(t == 0) continue;
      if(PositionGetInteger(POSITION_MAGIC) != (long)m_cfg.magic) continue;
      if(PositionGetString(POSITION_SYMBOL) != m_broker.Symbol())  continue;
      if(m_log != NULL)
         m_log.Warn("RISK", StringFormat("ปิดออเดอร์ #%I64u (%s)", t, why));
      m_broker.ClosePosition(t, m_cfg.deviation, 3);
     }
  }

#endif // __SNIPER_RISK_MQH__
