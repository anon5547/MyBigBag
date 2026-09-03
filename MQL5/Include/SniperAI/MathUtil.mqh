//+------------------------------------------------------------------+
//|                                                     MathUtil.mqh |
//|  SniperAI V5 - deterministic series math.                        |
//|                                                                  |
//|  EVERY function here has a bit-for-bit twin in                   |
//|  tools/sniper_features.py. If you change one, change both and    |
//|  re-run tools/parity_check.py, or your live features will drift  |
//|  away from your training features and the model becomes noise.   |
//|                                                                  |
//|  All arrays are CHRONOLOGICAL: index 0 = oldest bar.             |
//+------------------------------------------------------------------+
#property copyright "SniperAI V5"
#ifndef __SNIPER_MATHUTIL_MQH__
#define __SNIPER_MATHUTIL_MQH__

#define SNP_EPS 1e-9

//+------------------------------------------------------------------+
//| Safe division: never returns inf/nan.                            |
//+------------------------------------------------------------------+
double SnpDiv(const double a, const double b, const double fallback=0.0)
  {
   if(MathAbs(b) < SNP_EPS)
      return(fallback);
   double r = a / b;
   if(!MathIsValidNumber(r))
      return(fallback);
   return(r);
  }

//+------------------------------------------------------------------+
//| Clamp                                                            |
//+------------------------------------------------------------------+
double SnpClamp(const double v, const double lo, const double hi)
  {
   if(!MathIsValidNumber(v)) return(lo);
   if(v < lo) return(lo);
   if(v > hi) return(hi);
   return(v);
  }

//+------------------------------------------------------------------+
//| Simple moving average.                                           |
//| dst[i] for i < period-1 is back-filled with the partial mean so  |
//| the series never contains NaN (matches the Python reference).    |
//+------------------------------------------------------------------+
void SnpSMA(const double &src[], const int n, const int period, double &dst[])
  {
   ArrayResize(dst, n);
   if(n <= 0 || period <= 0) return;
   double run = 0.0;
   for(int i = 0; i < n; i++)
     {
      run += src[i];
      if(i >= period)
         run -= src[i - period];
      int cnt = (i + 1 < period) ? (i + 1) : period;
      dst[i] = run / cnt;
     }
  }

//+------------------------------------------------------------------+
//| Exponential MA, pandas style: ewm(span=period, adjust=False).    |
//| Seeded with src[0]. This is NOT MetaTrader's SMA-seeded EMA;     |
//| with 400 bars of warm-up the difference is < 1e-12.              |
//+------------------------------------------------------------------+
void SnpEMA(const double &src[], const int n, const int period, double &dst[])
  {
   ArrayResize(dst, n);
   if(n <= 0 || period <= 0) return;
   double a = 2.0 / (period + 1.0);
   dst[0] = src[0];
   for(int i = 1; i < n; i++)
      dst[i] = a * src[i] + (1.0 - a) * dst[i - 1];
  }

//+------------------------------------------------------------------+
//| Wilder smoothing (RMA) == ewm(alpha=1/period, adjust=False).     |
//| Used by RSI, ADX/DI and ATR.                                     |
//+------------------------------------------------------------------+
void SnpRMA(const double &src[], const int n, const int period, double &dst[])
  {
   ArrayResize(dst, n);
   if(n <= 0 || period <= 0) return;
   double a = 1.0 / (double)period;
   dst[0] = src[0];
   for(int i = 1; i < n; i++)
      dst[i] = a * src[i] + (1.0 - a) * dst[i - 1];
  }

//+------------------------------------------------------------------+
//| Rolling POPULATION standard deviation (ddof = 0).                |
//| pandas .rolling().std() defaults to ddof=1 - the Python twin     |
//| explicitly passes ddof=0 to stay in sync with this.              |
//+------------------------------------------------------------------+
void SnpRollingStd(const double &src[], const int n, const int period, double &dst[])
  {
   ArrayResize(dst, n);
   if(n <= 0 || period <= 0) return;
   double s = 0.0, s2 = 0.0;
   for(int i = 0; i < n; i++)
     {
      s  += src[i];
      s2 += src[i] * src[i];
      if(i >= period)
        {
         s  -= src[i - period];
         s2 -= src[i - period] * src[i - period];
        }
      int cnt = (i + 1 < period) ? (i + 1) : period;
      double mean = s / cnt;
      double var  = (s2 / cnt) - (mean * mean);
      if(var < 0.0) var = 0.0;          // float noise guard
      dst[i] = MathSqrt(var);
     }
  }

//+------------------------------------------------------------------+
//| Rolling maximum / minimum (window INCLUDES the current bar).     |
//| O(n*period); period here is <= 20 so it is irrelevant.           |
//+------------------------------------------------------------------+
void SnpRollingMax(const double &src[], const int n, const int period, double &dst[])
  {
   ArrayResize(dst, n);
   for(int i = 0; i < n; i++)
     {
      int start = i - period + 1;
      if(start < 0) start = 0;
      double m = src[start];
      for(int j = start + 1; j <= i; j++)
         if(src[j] > m) m = src[j];
      dst[i] = m;
     }
  }

void SnpRollingMin(const double &src[], const int n, const int period, double &dst[])
  {
   ArrayResize(dst, n);
   for(int i = 0; i < n; i++)
     {
      int start = i - period + 1;
      if(start < 0) start = 0;
      double m = src[start];
      for(int j = start + 1; j <= i; j++)
         if(src[j] < m) m = src[j];
      dst[i] = m;
     }
  }

//+------------------------------------------------------------------+
//| RSI, Wilder.                                                     |
//+------------------------------------------------------------------+
void SnpRSI(const double &close[], const int n, const int period, double &dst[])
  {
   ArrayResize(dst, n);
   if(n <= 0) return;
   double gain[], loss[], ag[], al[];
   ArrayResize(gain, n);
   ArrayResize(loss, n);
   gain[0] = 0.0;
   loss[0] = 0.0;
   for(int i = 1; i < n; i++)
     {
      double d = close[i] - close[i - 1];
      gain[i] = (d > 0.0) ?  d : 0.0;
      loss[i] = (d < 0.0) ? -d : 0.0;
     }
   SnpRMA(gain, n, period, ag);
   SnpRMA(loss, n, period, al);
   for(int i = 0; i < n; i++)
     {
      if(al[i] < SNP_EPS)
         dst[i] = (ag[i] < SNP_EPS) ? 50.0 : 100.0;
      else
        {
         double rs = ag[i] / al[i];
         dst[i] = 100.0 - (100.0 / (1.0 + rs));
        }
     }
  }

//+------------------------------------------------------------------+
//| True Range series.                                               |
//+------------------------------------------------------------------+
void SnpTrueRange(const double &high[], const double &low[], const double &close[],
                  const int n, double &dst[])
  {
   ArrayResize(dst, n);
   if(n <= 0) return;
   dst[0] = high[0] - low[0];
   for(int i = 1; i < n; i++)
     {
      double a = high[i] - low[i];
      double b = MathAbs(high[i] - close[i - 1]);
      double c = MathAbs(low[i]  - close[i - 1]);
      dst[i] = MathMax(a, MathMax(b, c));
     }
  }

//+------------------------------------------------------------------+
//| ADX / +DI / -DI, Wilder.                                         |
//+------------------------------------------------------------------+
void SnpADX(const double &high[], const double &low[], const double &close[],
            const int n, const int period,
            double &adx[], double &pdi[], double &mdi[])
  {
   ArrayResize(adx, n);
   ArrayResize(pdi, n);
   ArrayResize(mdi, n);
   if(n <= 0) return;

   double tr[], pdm[], mdm[], atr[], spdm[], smdm[], dx[];
   SnpTrueRange(high, low, close, n, tr);
   ArrayResize(pdm, n);
   ArrayResize(mdm, n);
   pdm[0] = 0.0;
   mdm[0] = 0.0;
   for(int i = 1; i < n; i++)
     {
      double up   = high[i] - high[i - 1];
      double down = low[i - 1] - low[i];
      pdm[i] = (up > down   && up   > 0.0) ? up   : 0.0;
      mdm[i] = (down > up   && down > 0.0) ? down : 0.0;
     }
   SnpRMA(tr,  n, period, atr);
   SnpRMA(pdm, n, period, spdm);
   SnpRMA(mdm, n, period, smdm);

   ArrayResize(dx, n);
   for(int i = 0; i < n; i++)
     {
      pdi[i] = 100.0 * SnpDiv(spdm[i], atr[i], 0.0);
      mdi[i] = 100.0 * SnpDiv(smdm[i], atr[i], 0.0);
      double sum = pdi[i] + mdi[i];
      dx[i] = 100.0 * SnpDiv(MathAbs(pdi[i] - mdi[i]), sum, 0.0);
     }
   SnpRMA(dx, n, period, adx);
  }

//+------------------------------------------------------------------+
//| Wilder ATR (RMA of True Range).                                  |
//| NOTE: MetaTrader's built-in iATR uses an SMA of TR. We do not     |
//| use iATR anywhere, precisely so both sides agree.                |
//+------------------------------------------------------------------+
void SnpATR(const double &high[], const double &low[], const double &close[],
            const int n, const int period, double &dst[])
  {
   double tr[];
   SnpTrueRange(high, low, close, n, tr);
   SnpRMA(tr, n, period, dst);
  }

#endif // __SNIPER_MATHUTIL_MQH__
