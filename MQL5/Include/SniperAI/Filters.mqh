//+------------------------------------------------------------------+
//|                                                      Filters.mqh |
//|  SniperAI V5 - the gates that sit between a signal and an order. |
//|                                                                  |
//|  A model trained on clean historical bars knows nothing about    |
//|  spread blow-outs, the 30 seconds around rollover, or the        |
//|  Sunday-open gap. These filters are where that knowledge lives.  |
//|  Deleting them will make your backtest look better and your      |
//|  account look worse.                                             |
//+------------------------------------------------------------------+
#property copyright "SniperAI V5"
#ifndef __SNIPER_FILTERS_MQH__
#define __SNIPER_FILTERS_MQH__

#include <SniperAI/MathUtil.mqh>

#define SNP_MAX_WINDOWS 8

struct SnpTimeWindow
  {
   int               from_min;   // minutes since server midnight
   int               to_min;
  };

//+------------------------------------------------------------------+
//| Parse "07:00-11:00,13:30-19:00" into windows.                    |
//| A window whose end is <= its start is treated as crossing        |
//| midnight (e.g. "22:00-02:00").                                   |
//+------------------------------------------------------------------+
int SnpParseWindows(const string spec, SnpTimeWindow &out[])
  {
   ArrayResize(out, 0);
   string s = spec;
   StringTrimLeft(s);
   StringTrimRight(s);
   if(s == "") return(0);

   //--- StringSplit wants a character CODE, not a char literal
   const ushort SEP_COMMA = StringGetCharacter(",", 0);
   const ushort SEP_DASH  = StringGetCharacter("-", 0);
   const ushort SEP_COLON = StringGetCharacter(":", 0);

   string parts[];
   int np = StringSplit(s, SEP_COMMA, parts);
   for(int i = 0; i < np && ArraySize(out) < SNP_MAX_WINDOWS; i++)
     {
      string p = parts[i];
      StringTrimLeft(p);
      StringTrimRight(p);
      if(p == "") continue;

      string ends[];
      if(StringSplit(p, SEP_DASH, ends) != 2) continue;

      string a[], b[];
      if(StringSplit(ends[0], SEP_COLON, a) != 2) continue;
      if(StringSplit(ends[1], SEP_COLON, b) != 2) continue;

      int f = (int)StringToInteger(a[0]) * 60 + (int)StringToInteger(a[1]);
      int t = (int)StringToInteger(b[0]) * 60 + (int)StringToInteger(b[1]);
      if(f < 0 || f > 1440 || t < 0 || t > 1440) continue;

      int n = ArraySize(out);
      ArrayResize(out, n + 1);
      out[n].from_min = f;
      out[n].to_min   = t;
     }
   return(ArraySize(out));
  }

//+------------------------------------------------------------------+
bool SnpInWindows(const SnpTimeWindow &wins[], const int minute_of_day)
  {
   int n = ArraySize(wins);
   if(n == 0) return(false);
   for(int i = 0; i < n; i++)
     {
      int f = wins[i].from_min, t = wins[i].to_min;
      if(t > f)
        {
         if(minute_of_day >= f && minute_of_day < t) return(true);
        }
      else
        {
         //--- crosses midnight
         if(minute_of_day >= f || minute_of_day < t) return(true);
        }
     }
   return(false);
  }

//+------------------------------------------------------------------+
struct SnpFilterConfig
  {
   bool              use_sessions;
   string            session_spec;        // server time
   string            blackout_spec;       // server time, daily recurring
   bool              trade_mon, trade_tue, trade_wed, trade_thu, trade_fri;
   int               friday_cutoff_min;   // stop opening after this (0 = off)
   int               rollover_guard_min;  // minutes either side of 00:00 to avoid

   int               max_spread_points;
   double            min_adx;             // 0 = off  (kill chop)
   double            max_adx;             // 0 = off  (kill exhausted trends)
   double            max_atr_ratio;       // 0 = off  (kill volatility spikes)
   double            min_atr_norm;        // 0 = off  (kill dead markets)
  };

//+------------------------------------------------------------------+
class CSnpFilters
  {
private:
   SnpFilterConfig   m_cfg;
   SnpTimeWindow     m_sessions[];
   SnpTimeWindow     m_blackouts[];
   string            m_last_reason;

public:
                     CSnpFilters() { m_last_reason = ""; }

   void              Configure(const SnpFilterConfig &cfg)
     {
      m_cfg = cfg;
      SnpParseWindows(m_cfg.session_spec,  m_sessions);
      SnpParseWindows(m_cfg.blackout_spec, m_blackouts);
     }

   int               SessionCount()  const { return(ArraySize(m_sessions));  }
   int               BlackoutCount() const { return(ArraySize(m_blackouts)); }
   string            LastReason()    const { return(m_last_reason);          }

   //--- time-of-day / day-of-week gate
   bool              TimeAllowed(const datetime server_now);
   //--- market-condition gate
   bool              RegimeAllowed(const double adx, const double atr_ratio, const double atr_norm);
   //--- execution-cost gate
   bool              SpreadAllowed(const int spread_points);
  };

//+------------------------------------------------------------------+
bool CSnpFilters::TimeAllowed(const datetime server_now)
  {
   m_last_reason = "";
   MqlDateTime dt;
   TimeToStruct(server_now, dt);
   int mod = dt.hour * 60 + dt.min;

   //--- weekend / weekday mask (MetaTrader: 0 = Sunday)
   bool day_ok = false;
   switch(dt.day_of_week)
     {
      case 1: day_ok = m_cfg.trade_mon; break;
      case 2: day_ok = m_cfg.trade_tue; break;
      case 3: day_ok = m_cfg.trade_wed; break;
      case 4: day_ok = m_cfg.trade_thu; break;
      case 5: day_ok = m_cfg.trade_fri; break;
      default: day_ok = false;          // Sat/Sun: the Sunday open gap is not a trade
     }
   if(!day_ok)
     {
      m_last_reason = "นอกวันที่อนุญาตให้เทรด";
      return(false);
     }

   if(dt.day_of_week == 5 && m_cfg.friday_cutoff_min > 0 && mod >= m_cfg.friday_cutoff_min)
     {
      m_last_reason = "เลยเวลาปิดของวันศุกร์";
      return(false);
     }

   if(m_cfg.rollover_guard_min > 0)
     {
      int dist = MathMin(mod, 1440 - mod);
      if(dist < m_cfg.rollover_guard_min)
        {
         m_last_reason = "อยู่ในช่วง rollover (สเปรดกว้าง)";
         return(false);
        }
     }

   if(ArraySize(m_blackouts) > 0 && SnpInWindows(m_blackouts, mod))
     {
      m_last_reason = "อยู่ในช่วง blackout (ข่าว)";
      return(false);
     }

   if(m_cfg.use_sessions)
     {
      if(ArraySize(m_sessions) == 0)
        {
         m_last_reason = "เปิดกรองเซสชันแต่ตั้งค่าเวลาไม่ถูกต้อง";
         return(false);
        }
      if(!SnpInWindows(m_sessions, mod))
        {
         m_last_reason = "นอกช่วงเวลาเทรด";
         return(false);
        }
     }
   return(true);
  }

//+------------------------------------------------------------------+
bool CSnpFilters::RegimeAllowed(const double adx, const double atr_ratio, const double atr_norm)
  {
   m_last_reason = "";
   if(m_cfg.min_adx > 0.0 && adx < m_cfg.min_adx)
     {
      m_last_reason = StringFormat("ตลาด sideway (ADX %.1f < %.1f)", adx, m_cfg.min_adx);
      return(false);
     }
   if(m_cfg.max_adx > 0.0 && adx > m_cfg.max_adx)
     {
      m_last_reason = StringFormat("เทรนด์ยืดเกินไป (ADX %.1f > %.1f)", adx, m_cfg.max_adx);
      return(false);
     }
   if(m_cfg.max_atr_ratio > 0.0 && atr_ratio > m_cfg.max_atr_ratio)
     {
      m_last_reason = StringFormat("ความผันผวนพุ่ง (ATR ratio %.2f > %.2f)", atr_ratio, m_cfg.max_atr_ratio);
      return(false);
     }
   if(m_cfg.min_atr_norm > 0.0 && atr_norm < m_cfg.min_atr_norm)
     {
      m_last_reason = StringFormat("ตลาดนิ่งเกินไป (ATR%% %.4f < %.4f)", atr_norm, m_cfg.min_atr_norm);
      return(false);
     }
   return(true);
  }

//+------------------------------------------------------------------+
bool CSnpFilters::SpreadAllowed(const int spread_points)
  {
   m_last_reason = "";
   if(m_cfg.max_spread_points <= 0) return(true);
   if(spread_points > m_cfg.max_spread_points)
     {
      m_last_reason = StringFormat("สเปรดกว้างเกินไป (%d > %d จุด)", spread_points, m_cfg.max_spread_points);
      return(false);
     }
   return(true);
  }

#endif // __SNIPER_FILTERS_MQH__
