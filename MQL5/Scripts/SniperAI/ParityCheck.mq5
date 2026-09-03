//+------------------------------------------------------------------+
//|                                                  ParityCheck.mq5 |
//|  Dumps (a) the raw bars the EA would see and (b) the feature      |
//|  vectors it computes from them, so the offline Python trainer can |
//|  be proved identical to the live MQL5 code.                       |
//|                                                                   |
//|  Run this on the chart you intend to trade, then:                 |
//|                                                                   |
//|    python3 tools/parity_check.py \                                |
//|        --csv  <terminal>/MQL5/Files/SniperAI/parity_bars.csv \    |
//|        --mql5 <terminal>/MQL5/Files/SniperAI/parity_mql5.csv      |
//|                                                                   |
//|  If that prints PARITY OK, your training features and your live    |
//|  features are the same thing. If it does not, nothing else in      |
//|  this project matters until it does.                              |
//+------------------------------------------------------------------+
#property copyright "SniperAI V5"
#property version   "5.00"
#property script_show_inputs

#include <SniperAI/Features.mqh>

input ENUM_TIMEFRAMES InpTimeframe = PERIOD_M5;   // Timeframe
input int  InpTotalBars   = 2500;                 // Bars to export
input int  InpProbeCount  = 60;                   // Feature vectors to dump
input int  InpWindow      = 1200;                 // Rolling window per probe (>= 1000)

//+------------------------------------------------------------------+
void OnStart()
  {
   if(InpWindow < SNP_WARMUP_BARS)
     {
      Print("InpWindow must be >= ", SNP_WARMUP_BARS);
      return;
     }
   if(InpTotalBars < InpWindow + InpProbeCount)
     {
      Print("InpTotalBars is too small for that window / probe count");
      return;
     }

   MqlRates rates[];
   ArraySetAsSeries(rates, false);
   int copied = CopyRates(_Symbol, InpTimeframe, 1, InpTotalBars, rates);
   if(copied < InpWindow + InpProbeCount)
     {
      Print("CopyRates returned only ", copied, " bars. Scroll the chart back to load history.");
      return;
     }
   PrintFormat("copied %d bars, %s %s", copied, _Symbol, EnumToString(InpTimeframe));

   //=== 1. the raw bars ============================================
   int fb = FileOpen("SniperAI\\parity_bars.csv", FILE_WRITE | FILE_TXT | FILE_ANSI);
   if(fb == INVALID_HANDLE)
     {
      Print("cannot write parity_bars.csv, err=", GetLastError());
      return;
     }
   FileWriteString(fb, "time,open,high,low,close,tick_volume\r\n");
   for(int i = 0; i < copied; i++)
     {
      FileWriteString(fb, StringFormat("%s,%.8f,%.8f,%.8f,%.8f,%.1f\r\n",
                                       TimeToString(rates[i].time, TIME_DATE | TIME_SECONDS),
                                       rates[i].open, rates[i].high, rates[i].low, rates[i].close,
                                       (double)rates[i].tick_volume));
     }
   FileClose(fb);

   //=== 2. the feature vectors =====================================
   int ff = FileOpen("SniperAI\\parity_mql5.csv", FILE_WRITE | FILE_TXT | FILE_ANSI);
   if(ff == INVALID_HANDLE)
     {
      Print("cannot write parity_mql5.csv, err=", GetLastError());
      return;
     }
   string hdr = "bar_time";
   for(int k = 0; k < SNP_FEATURE_DIM; k++)
      hdr += "," + SnpFeatureName(k);
   FileWriteString(ff, hdr + "\r\n");

   CSnpFeatureEngine eng;
   int first = InpWindow - 1;
   int last  = copied - 1;
   int step  = MathMax(1, (last - first) / MathMax(1, InpProbeCount - 1));
   int dumped = 0, failed = 0;

   for(int end = first; end <= last; end += step)
     {
      int start = end - InpWindow + 1;
      MqlRates win[];
      ArrayResize(win, InpWindow);
      for(int j = 0; j < InpWindow; j++)
         win[j] = rates[start + j];

      float f[];
      if(!eng.Build(win, InpWindow, f))
        {
         PrintFormat("probe at %s failed: %s",
                     TimeToString(rates[end].time, TIME_DATE | TIME_SECONDS), eng.Error());
         failed++;
         continue;
        }

      string row = TimeToString(rates[end].time, TIME_DATE | TIME_SECONDS);
      for(int k = 0; k < SNP_FEATURE_DIM; k++)
         row += StringFormat(",%.9g", (double)f[k]);
      FileWriteString(ff, row + "\r\n");
      dumped++;
     }
   FileClose(ff);

   PrintFormat("wrote %d feature rows (%d failed) to MQL5\\Files\\SniperAI\\parity_mql5.csv", dumped, failed);
   PrintFormat("wrote %d bars to MQL5\\Files\\SniperAI\\parity_bars.csv", copied);
   Print("now run: python3 tools/parity_check.py --csv parity_bars.csv --mql5 parity_mql5.csv");
  }
//+------------------------------------------------------------------+
