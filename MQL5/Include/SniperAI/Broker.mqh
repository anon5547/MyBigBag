//+------------------------------------------------------------------+
//|                                                       Broker.mqh |
//|  SniperAI V5 - symbol normalisation and order execution.         |
//|                                                                  |
//|  Everything a broker can use to reject or silently mangle your   |
//|  order lives here: lot step, stops level, freeze level, filling  |
//|  mode, spread, digits, and the retryable retcodes.               |
//|                                                                  |
//|  Money -> price conversion uses OrderCalcProfit, NOT a hand      |
//|  rolled tick-value formula, so it stays correct on accounts      |
//|  whose deposit currency differs from the quote currency.         |
//+------------------------------------------------------------------+
#property copyright "SniperAI V5"
#ifndef __SNIPER_BROKER_MQH__
#define __SNIPER_BROKER_MQH__

#include <SniperAI/MathUtil.mqh>
#include <SniperAI/Logger.mqh>

struct SnpOrderResult
  {
   bool              ok;
   ulong             ticket;
   double            price;
   double            volume;
   uint              retcode;
   string            message;
  };

//+------------------------------------------------------------------+
class CSnpBroker
  {
private:
   string            m_symbol;
   int               m_digits;
   double            m_point;
   double            m_lot_min, m_lot_max, m_lot_step;
   int               m_stops_level;      // points
   int               m_freeze_level;     // points
   ENUM_ORDER_TYPE_FILLING m_filling;
   CSnpLogger       *m_log;

   bool              IsRetryable(const uint rc) const;

public:
                     CSnpBroker() { m_log = NULL; m_symbol = ""; }

   bool              Init(const string sym, CSnpLogger *logger);
   void              Refresh();

   string            Symbol()      const { return(m_symbol);      }
   int               Digits()      const { return(m_digits);      }
   double            Point()       const { return(m_point);       }
   double            LotMin()      const { return(m_lot_min);     }
   double            LotMax()      const { return(m_lot_max);     }
   int               StopsLevel()  const { return(m_stops_level); }
   ENUM_ORDER_TYPE_FILLING Filling() const { return(m_filling);   }

   double            Ask() const { return(SymbolInfoDouble(m_symbol, SYMBOL_ASK)); }
   double            Bid() const { return(SymbolInfoDouble(m_symbol, SYMBOL_BID)); }
   int               SpreadPoints() const;

   double            NormalizeLot(const double lot) const;
   double            NormalizePrice(const double p) const { return(NormalizeDouble(p, m_digits)); }

   //--- money <-> price distance, for a given volume
   double            MoneyPerPriceUnit(const double lot) const;
   double            MoneyToPriceDistance(const double money, const double lot) const;
   double            PriceDistanceToMoney(const double dist, const double lot) const;

   //--- broker-legal stop placement
   double            MinStopDistance() const { return(m_stops_level * m_point); }
   bool              ClampStops(const ENUM_ORDER_TYPE type, const double entry,
                                double &sl, double &tp) const;

   bool              MarketOpen() const;

   SnpOrderResult    Open(const ENUM_ORDER_TYPE type, const double lot,
                          const double sl, const double tp,
                          const ulong magic, const string comment,
                          const int deviation, const int max_retries);
   bool              ModifyStops(const ulong ticket, const double sl, const double tp);
   bool              ClosePosition(const ulong ticket, const int deviation, const int max_retries);
  };

//+------------------------------------------------------------------+
bool CSnpBroker::Init(const string sym, CSnpLogger *logger)
  {
   m_log    = logger;
   m_symbol = (sym == "") ? _Symbol : sym;

   if(!SymbolInfoInteger(m_symbol, SYMBOL_SELECT))
     {
      if(!SymbolSelect(m_symbol, true))
        {
         if(m_log != NULL) m_log.Error("BROKER", "เลือกสัญลักษณ์ไม่ได้: " + m_symbol);
         return(false);
        }
     }
   Refresh();

   if(m_lot_step <= 0.0 || m_lot_min <= 0.0)
     {
      if(m_log != NULL) m_log.Error("BROKER", "ข้อมูล lot ของสัญลักษณ์ผิดปกติ");
      return(false);
     }
   return(true);
  }

//+------------------------------------------------------------------+
void CSnpBroker::Refresh()
  {
   m_digits       = (int)SymbolInfoInteger(m_symbol, SYMBOL_DIGITS);
   m_point        = SymbolInfoDouble(m_symbol, SYMBOL_POINT);
   m_lot_min      = SymbolInfoDouble(m_symbol, SYMBOL_VOLUME_MIN);
   m_lot_max      = SymbolInfoDouble(m_symbol, SYMBOL_VOLUME_MAX);
   m_lot_step     = SymbolInfoDouble(m_symbol, SYMBOL_VOLUME_STEP);
   m_stops_level  = (int)SymbolInfoInteger(m_symbol, SYMBOL_TRADE_STOPS_LEVEL);
   m_freeze_level = (int)SymbolInfoInteger(m_symbol, SYMBOL_TRADE_FREEZE_LEVEL);

   //--- pick a filling mode the broker actually advertises
   int modes = (int)SymbolInfoInteger(m_symbol, SYMBOL_FILLING_MODE);
   if((modes & SYMBOL_FILLING_FOK) != 0)
      m_filling = ORDER_FILLING_FOK;
   else if((modes & SYMBOL_FILLING_IOC) != 0)
      m_filling = ORDER_FILLING_IOC;
   else
      m_filling = ORDER_FILLING_RETURN;
  }

//+------------------------------------------------------------------+
int CSnpBroker::SpreadPoints() const
  {
   double a = Ask(), b = Bid();
   if(a <= 0.0 || b <= 0.0 || m_point <= 0.0) return(INT_MAX);
   return((int)MathRound((a - b) / m_point));
  }

//+------------------------------------------------------------------+
double CSnpBroker::NormalizeLot(const double lot) const
  {
   if(m_lot_step <= 0.0) return(0.0);
   double v = MathFloor(lot / m_lot_step + 1e-8) * m_lot_step;
   if(v < m_lot_min) v = m_lot_min;
   if(v > m_lot_max) v = m_lot_max;
   //--- kill float dust: 0.0300000000004 is rejected by some servers
   int step_digits = (int)MathMax(0.0, MathCeil(-MathLog10(m_lot_step) - 1e-9));
   return(NormalizeDouble(v, step_digits));
  }

//+------------------------------------------------------------------+
//| Account-currency profit for a +1.0 move of the raw price, at the |
//| given volume. Falls back to tick maths if OrderCalcProfit is     |
//| unavailable (some testers / exotic symbols).                     |
//+------------------------------------------------------------------+
double CSnpBroker::MoneyPerPriceUnit(const double lot) const
  {
   double price = Ask();
   double profit = 0.0;
   if(price > 0.0 &&
      OrderCalcProfit(ORDER_TYPE_BUY, m_symbol, lot, price, price + 1.0, profit) &&
      MathAbs(profit) > SNP_EPS)
      return(MathAbs(profit));

   double tv = SymbolInfoDouble(m_symbol, SYMBOL_TRADE_TICK_VALUE);
   double ts = SymbolInfoDouble(m_symbol, SYMBOL_TRADE_TICK_SIZE);
   if(tv > 0.0 && ts > 0.0)
      return(lot * tv / ts);
   return(0.0);
  }

//+------------------------------------------------------------------+
double CSnpBroker::MoneyToPriceDistance(const double money, const double lot) const
  {
   double mppu = MoneyPerPriceUnit(lot);
   if(mppu <= SNP_EPS) return(0.0);
   return(money / mppu);
  }

double CSnpBroker::PriceDistanceToMoney(const double dist, const double lot) const
  {
   return(dist * MoneyPerPriceUnit(lot));
  }

//+------------------------------------------------------------------+
//| Push SL/TP outside the broker's minimum stop distance.           |
//| Returns false if a stop had to be pushed so far it no longer     |
//| means what the caller asked for - caller decides what to do.     |
//+------------------------------------------------------------------+
bool CSnpBroker::ClampStops(const ENUM_ORDER_TYPE type, const double entry,
                            double &sl, double &tp) const
  {
   double mind = MinStopDistance();
   if(mind <= 0.0) mind = 2 * m_point;      // never allow a zero-distance stop
   bool clean = true;

   if(type == ORDER_TYPE_BUY)
     {
      if(sl > 0.0 && entry - sl < mind) { sl = entry - mind; clean = false; }
      if(tp > 0.0 && tp - entry < mind) { tp = entry + mind; clean = false; }
     }
   else
     {
      if(sl > 0.0 && sl - entry < mind) { sl = entry + mind; clean = false; }
      if(tp > 0.0 && entry - tp < mind) { tp = entry - mind; clean = false; }
     }
   if(sl > 0.0) sl = NormalizePrice(sl);
   if(tp > 0.0) tp = NormalizePrice(tp);
   return(clean);
  }

//+------------------------------------------------------------------+
bool CSnpBroker::MarketOpen() const
  {
   if(!SymbolInfoInteger(m_symbol, SYMBOL_TRADE_MODE))
      return(false);
   ENUM_SYMBOL_TRADE_MODE tm = (ENUM_SYMBOL_TRADE_MODE)SymbolInfoInteger(m_symbol, SYMBOL_TRADE_MODE);
   if(tm != SYMBOL_TRADE_MODE_FULL && tm != SYMBOL_TRADE_MODE_LONGONLY && tm != SYMBOL_TRADE_MODE_SHORTONLY)
      return(false);
   if(Ask() <= 0.0 || Bid() <= 0.0)
      return(false);
   return(true);
  }

//+------------------------------------------------------------------+
bool CSnpBroker::IsRetryable(const uint rc) const
  {
   return(rc == TRADE_RETCODE_REQUOTE        ||
          rc == TRADE_RETCODE_PRICE_CHANGED  ||
          rc == TRADE_RETCODE_PRICE_OFF      ||
          rc == TRADE_RETCODE_TIMEOUT        ||
          rc == TRADE_RETCODE_CONNECTION     ||
          rc == TRADE_RETCODE_TOO_MANY_REQUESTS);
  }

//+------------------------------------------------------------------+
SnpOrderResult CSnpBroker::Open(const ENUM_ORDER_TYPE type, const double lot,
                                const double sl, const double tp,
                                const ulong magic, const string comment,
                                const int deviation, const int max_retries)
  {
   SnpOrderResult r;
   r.ok = false; r.ticket = 0; r.price = 0.0; r.volume = 0.0;
   r.retcode = 0; r.message = "";

   double vol = NormalizeLot(lot);
   if(vol <= 0.0)
     {
      r.message = "lot ไม่ถูกต้อง";
      return(r);
     }

   for(int attempt = 0; attempt <= max_retries; attempt++)
     {
      MqlTradeRequest req;
      MqlTradeResult  res;
      ZeroMemory(req);
      ZeroMemory(res);

      double price = (type == ORDER_TYPE_BUY) ? Ask() : Bid();
      if(price <= 0.0)
        {
         r.message = "ไม่มีราคา (ตลาดปิดหรือขาดการเชื่อมต่อ)";
         return(r);
        }

      double use_sl = sl, use_tp = tp;
      ClampStops(type, price, use_sl, use_tp);

      req.action       = TRADE_ACTION_DEAL;
      req.symbol       = m_symbol;
      req.volume       = vol;
      req.type         = type;
      req.price        = NormalizePrice(price);
      req.sl           = (use_sl > 0.0) ? use_sl : 0.0;
      req.tp           = (use_tp > 0.0) ? use_tp : 0.0;
      req.deviation    = (ulong)MathMax(1, deviation);
      req.magic        = magic;
      req.comment      = comment;
      req.type_filling = m_filling;
      req.type_time    = ORDER_TIME_GTC;

      MqlTradeCheckResult chk;
      ZeroMemory(chk);
      if(!OrderCheck(req, chk))
        {
         r.retcode = chk.retcode;
         r.message = StringFormat("OrderCheck ปฏิเสธ rc=%u %s", chk.retcode, chk.comment);

         //--- a filling mode the server does not accept is not a real
         //--- rejection; cycle to the next one and re-check.
         if(chk.retcode == TRADE_RETCODE_INVALID_FILL && attempt < max_retries)
           {
            if(m_filling == ORDER_FILLING_FOK)      m_filling = ORDER_FILLING_IOC;
            else if(m_filling == ORDER_FILLING_IOC) m_filling = ORDER_FILLING_RETURN;
            else                                    { if(m_log != NULL) m_log.Error("MT5", r.message); return(r); }
            if(m_log != NULL) m_log.Warn("MT5", "OrderCheck: สลับ filling mode แล้วลองใหม่");
            continue;
           }

         //--- anything else (margin, volume, stops) will not fix itself
         if(m_log != NULL) m_log.Error("MT5", r.message);
         return(r);
        }

      bool sent = OrderSend(req, res);
      r.retcode  = res.retcode;

      if(sent && (res.retcode == TRADE_RETCODE_DONE ||
                  res.retcode == TRADE_RETCODE_DONE_PARTIAL ||
                  res.retcode == TRADE_RETCODE_PLACED))
        {
         r.ok     = true;
         r.ticket = (res.order > 0) ? res.order : res.deal;
         r.price  = (res.price > 0.0) ? res.price : req.price;
         r.volume = (res.volume > 0.0) ? res.volume : vol;
         return(r);
        }

      r.message = StringFormat("OrderSend rc=%u (%s)", res.retcode, res.comment);

      //--- some servers reject the filling mode only at send time
      if(res.retcode == TRADE_RETCODE_INVALID_FILL)
        {
         if(m_filling == ORDER_FILLING_FOK)      m_filling = ORDER_FILLING_IOC;
         else if(m_filling == ORDER_FILLING_IOC) m_filling = ORDER_FILLING_RETURN;
         else                                    return(r);
         if(m_log != NULL) m_log.Warn("MT5", "สลับ filling mode แล้วลองใหม่");
         continue;
        }

      if(!IsRetryable(res.retcode))
        {
         if(m_log != NULL) m_log.Error("MT5", r.message);
         return(r);
        }

      if(m_log != NULL)
         m_log.Warn("MT5", StringFormat("%s | ลองใหม่ครั้งที่ %d", r.message, attempt + 1));
      Sleep(200 * (attempt + 1));
      Refresh();
     }

   if(m_log != NULL) m_log.Error("MT5", "ส่งออเดอร์ไม่สำเร็จหลังจากลองใหม่ครบแล้ว");
   return(r);
  }

//+------------------------------------------------------------------+
bool CSnpBroker::ModifyStops(const ulong ticket, const double sl, const double tp)
  {
   if(!PositionSelectByTicket(ticket))
      return(false);

   double cur_sl = PositionGetDouble(POSITION_SL);
   double cur_tp = PositionGetDouble(POSITION_TP);
   double new_sl = (sl > 0.0) ? NormalizePrice(sl) : 0.0;
   double new_tp = (tp > 0.0) ? NormalizePrice(tp) : 0.0;

   //--- nothing to do: never spam the server with identical modifies
   if(MathAbs(cur_sl - new_sl) < m_point * 0.5 && MathAbs(cur_tp - new_tp) < m_point * 0.5)
      return(true);

   //--- freeze level: the server refuses changes this close to price
   if(m_freeze_level > 0)
     {
      double px   = PositionGetDouble(POSITION_PRICE_CURRENT);
      double frz  = m_freeze_level * m_point;
      if((new_sl > 0.0 && MathAbs(px - new_sl) < frz) ||
         (new_tp > 0.0 && MathAbs(px - new_tp) < frz))
        {
         if(m_log != NULL) m_log.Debug("MT5", "อยู่ในระยะ freeze level ข้ามการแก้ไข");
         return(false);
        }
     }

   MqlTradeRequest req;
   MqlTradeResult  res;
   ZeroMemory(req);
   ZeroMemory(res);
   req.action   = TRADE_ACTION_SLTP;
   req.symbol   = m_symbol;
   req.position = ticket;
   req.sl       = new_sl;
   req.tp       = new_tp;

   if(!OrderSend(req, res) || res.retcode != TRADE_RETCODE_DONE)
     {
      if(m_log != NULL)
         m_log.Warn("MT5", StringFormat("แก้ SL/TP ไม่สำเร็จ #%I64u rc=%u %s", ticket, res.retcode, res.comment));
      return(false);
     }
   return(true);
  }

//+------------------------------------------------------------------+
bool CSnpBroker::ClosePosition(const ulong ticket, const int deviation, const int max_retries)
  {
   for(int attempt = 0; attempt <= max_retries; attempt++)
     {
      if(!PositionSelectByTicket(ticket))
         return(true);                       // already gone

      ENUM_POSITION_TYPE ptype = (ENUM_POSITION_TYPE)PositionGetInteger(POSITION_TYPE);
      double vol = PositionGetDouble(POSITION_VOLUME);

      MqlTradeRequest req;
      MqlTradeResult  res;
      ZeroMemory(req);
      ZeroMemory(res);
      req.action       = TRADE_ACTION_DEAL;
      req.symbol       = PositionGetString(POSITION_SYMBOL);
      req.position     = ticket;
      req.volume       = vol;
      req.type         = (ptype == POSITION_TYPE_BUY) ? ORDER_TYPE_SELL : ORDER_TYPE_BUY;
      req.price        = (ptype == POSITION_TYPE_BUY) ? Bid() : Ask();
      req.deviation    = (ulong)MathMax(1, deviation);
      req.magic        = PositionGetInteger(POSITION_MAGIC);
      req.type_filling = m_filling;

      if(OrderSend(req, res) && res.retcode == TRADE_RETCODE_DONE)
         return(true);

      if(res.retcode == TRADE_RETCODE_INVALID_FILL)
        {
         if(m_filling == ORDER_FILLING_FOK)      m_filling = ORDER_FILLING_IOC;
         else if(m_filling == ORDER_FILLING_IOC) m_filling = ORDER_FILLING_RETURN;
         continue;
        }
      if(!IsRetryable(res.retcode))
        {
         if(m_log != NULL)
            m_log.Error("MT5", StringFormat("ปิดออเดอร์ #%I64u ไม่สำเร็จ rc=%u %s", ticket, res.retcode, res.comment));
         return(false);
        }
      Sleep(200 * (attempt + 1));
     }
   return(false);
  }

#endif // __SNIPER_BROKER_MQH__
