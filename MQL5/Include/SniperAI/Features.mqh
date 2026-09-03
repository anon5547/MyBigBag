//+------------------------------------------------------------------+
//|                                                     Features.mqh |
//|  SniperAI V5 - the 40-dimension feature vector.                  |
//|                                                                  |
//|  CONTRACT: tools/feature_contract.json                           |
//|  PYTHON TWIN: tools/sniper_features.py                           |
//|  PARITY TEST: MQL5/Scripts/SniperAI/ParityCheck.mq5              |
//|                                                                  |
//|  Rule of the house: a feature that cannot be reproduced          |
//|  identically offline is not a feature, it is a bug with a name.  |
//|  That is why nothing here uses spread, account state, or any     |
//|  broker-specific value - only OHLCV + bar time.                  |
//+------------------------------------------------------------------+
#property copyright "SniperAI V5"
#ifndef __SNIPER_FEATURES_MQH__
#define __SNIPER_FEATURES_MQH__

#include <SniperAI/MathUtil.mqh>

#define SNP_FEATURE_DIM   40
#define SNP_META_DIM      43
#define SNP_WARMUP_BARS 1000

//--- feature index constants (keep in lockstep with the contract) ---
enum ENUM_SNP_FEATURE
  {
   F_RET_1=0, F_RET_3, F_RET_5, F_RET_10,
   F_RET_1_LAG1, F_RET_1_LAG2, F_RET_1_LAG3,
   F_RSI_7, F_RSI_14, F_RSI_14_LAG1,
   F_MACD, F_MACD_SIGNAL, F_MACD_HIST, F_MACD_HIST_LAG1,
   F_ADX_14, F_PLUS_DI_14, F_MINUS_DI_14,
   F_EMA20_DIST, F_SMA60_DIST, F_EMA20_SLOPE,
   F_ATR14_NORM, F_ATR_RATIO,
   F_BB_PCTB, F_BB_WIDTH,
   F_DIST_TO_RES, F_DIST_TO_SUP, F_DONCHIAN_WIDTH, F_CLOSE_ZSCORE_20,
   F_RANGE_PCT, F_BODY_PCT, F_UPPER_WICK_PCT, F_LOWER_WICK_PCT,
   F_VOL_DELTA, F_RVOL_20,
   F_STOCH_K_14, F_STOCH_D_3,
   F_HOUR_SIN, F_HOUR_COS, F_DOW_SIN, F_DOW_COS
  };

string SnpFeatureName(const int i)
  {
   static string names[SNP_FEATURE_DIM] =
     {
      "ret_1","ret_3","ret_5","ret_10",
      "ret_1_lag1","ret_1_lag2","ret_1_lag3",
      "rsi_7","rsi_14","rsi_14_lag1",
      "macd","macd_signal","macd_hist","macd_hist_lag1",
      "adx_14","plus_di_14","minus_di_14",
      "ema20_dist","sma60_dist","ema20_slope",
      "atr14_norm","atr_ratio",
      "bb_pctb","bb_width",
      "dist_to_res","dist_to_sup","donchian_width","close_zscore_20",
      "range_pct","body_pct","upper_wick_pct","lower_wick_pct",
      "vol_delta","rvol_20",
      "stoch_k_14","stoch_d_3",
      "hour_sin","hour_cos","dow_sin","dow_cos"
     };
   if(i < 0 || i >= SNP_FEATURE_DIM) return("?");
   return(names[i]);
  }

//+------------------------------------------------------------------+
//| CSnpFeatureEngine                                                |
//|                                                                  |
//| Build(rates, count, out[]) -> fills SNP_FEATURE_DIM floats for   |
//| the LAST element of the rates array. Caller is responsible for   |
//| only passing CLOSED bars.                                        |
//+------------------------------------------------------------------+
class CSnpFeatureEngine
  {
private:
   //--- raw
   double            m_open[], m_high[], m_low[], m_close[], m_vol[];
   datetime          m_time[];
   int               m_n;
   //--- last diagnostics, exposed for the panel / logs
   double            m_last_atr;
   double            m_last_adx;
   double            m_last_rsi14;
   string            m_error;

public:
                     CSnpFeatureEngine() { m_n = 0; m_last_atr = 0; m_last_adx = 0; m_last_rsi14 = 50; m_error = ""; }

   double            LastATR()    const { return(m_last_atr);   }
   double            LastADX()    const { return(m_last_adx);   }
   double            LastRSI14()  const { return(m_last_rsi14); }
   string            Error()      const { return(m_error);      }

   bool              Build(const MqlRates &rates[], const int count, float &out[]);
   bool              BuildFromArrays(const double &o[], const double &h[], const double &l[],
                                     const double &c[], const double &v[], const datetime &t[],
                                     const int count, float &out[]);
   //--- meta vector = 40 base features + [p_up, side, edge]
   static void       BuildMeta(const float &base[], const double p_up, const double side, float &meta[]);
  };

//+------------------------------------------------------------------+
bool CSnpFeatureEngine::Build(const MqlRates &rates[], const int count, float &out[])
  {
   if(count < SNP_WARMUP_BARS)
     {
      m_error = StringFormat("need %d bars, got %d", SNP_WARMUP_BARS, count);
      return(false);
     }
   double o[], h[], l[], c[], v[];
   datetime t[];
   ArrayResize(o, count); ArrayResize(h, count); ArrayResize(l, count);
   ArrayResize(c, count); ArrayResize(v, count); ArrayResize(t, count);
   for(int i = 0; i < count; i++)
     {
      o[i] = rates[i].open;
      h[i] = rates[i].high;
      l[i] = rates[i].low;
      c[i] = rates[i].close;
      v[i] = (double)rates[i].tick_volume;
      t[i] = rates[i].time;
     }
   return(BuildFromArrays(o, h, l, c, v, t, count, out));
  }

//+------------------------------------------------------------------+
bool CSnpFeatureEngine::BuildFromArrays(const double &o[], const double &h[], const double &l[],
                                        const double &c[], const double &v[], const datetime &t[],
                                        const int count, float &out[])
  {
   m_error = "";
   if(count < 70)   // absolute minimum for sma60 + lags
     {
      m_error = "not enough bars";
      return(false);
     }
   const int n = count;
   const int i = n - 1;             // the bar we are describing

   ArrayResize(out, SNP_FEATURE_DIM);
   ArrayInitialize(out, 0.0f);

   //--- sanity: reject a corrupt window rather than feeding the model garbage
   for(int k = n - 5; k < n; k++)
     {
      if(c[k] <= 0.0 || h[k] < l[k] || !MathIsValidNumber(c[k]))
        {
         m_error = StringFormat("corrupt bar at %d", k);
         return(false);
        }
     }

   //================= returns ======================================
   double f_ret1  = MathLog(SnpDiv(c[i],     c[i-1],  1.0));
   double f_ret3  = MathLog(SnpDiv(c[i],     c[i-3],  1.0));
   double f_ret5  = MathLog(SnpDiv(c[i],     c[i-5],  1.0));
   double f_ret10 = MathLog(SnpDiv(c[i],     c[i-10], 1.0));
   double f_ret1_l1 = MathLog(SnpDiv(c[i-1], c[i-2],  1.0));
   double f_ret1_l2 = MathLog(SnpDiv(c[i-2], c[i-3],  1.0));
   double f_ret1_l3 = MathLog(SnpDiv(c[i-3], c[i-4],  1.0));

   //================= momentum ====================================
   double rsi7[], rsi14[];
   SnpRSI(c, n, 7,  rsi7);
   SnpRSI(c, n, 14, rsi14);

   double ema12[], ema26[], macd[], macdsig[];
   SnpEMA(c, n, 12, ema12);
   SnpEMA(c, n, 26, ema26);
   ArrayResize(macd, n);
   for(int k = 0; k < n; k++)
      macd[k] = ema12[k] - ema26[k];
   SnpEMA(macd, n, 9, macdsig);

   //--- MACD is normalised by price. XAUUSD went from 1800 to 4300 in
   //--- this dataset; a raw MACD value is a different animal at each
   //--- price level and is the classic silent killer of gold models.
   double macd_n     = 1000.0 * SnpDiv(macd[i],                     c[i]);
   double macdsig_n  = 1000.0 * SnpDiv(macdsig[i],                  c[i]);
   double macdhist_n = 1000.0 * SnpDiv(macd[i]   - macdsig[i],      c[i]);
   double macdhist_l1= 1000.0 * SnpDiv(macd[i-1] - macdsig[i-1],    c[i-1]);

   //================= trend =======================================
   double adx[], pdi[], mdi[];
   SnpADX(h, l, c, n, 14, adx, pdi, mdi);

   double ema20[], sma60[];
   SnpEMA(c, n, 20, ema20);
   SnpSMA(c, n, 60, sma60);

   double ema20_dist  = SnpDiv(c[i] - ema20[i], c[i]);
   double sma60_dist  = SnpDiv(c[i] - sma60[i], c[i]);
   double ema20_slope = SnpDiv(ema20[i] - ema20[i-5], c[i]);

   //================= volatility ==================================
   double atr14[], atr50[];
   SnpATR(h, l, c, n, 14, atr14);
   SnpATR(h, l, c, n, 50, atr50);
   double atr14_norm = SnpDiv(atr14[i], c[i]);
   double atr_ratio  = SnpDiv(atr14[i], atr50[i], 1.0);

   double sd20[];
   SnpRollingStd(c, n, 20, sd20);
   double sma20[];
   SnpSMA(c, n, 20, sma20);
   double bb_up  = sma20[i] + 2.0 * sd20[i];
   double bb_dn  = sma20[i] - 2.0 * sd20[i];
   double bb_pctb  = SnpDiv(c[i] - bb_dn, bb_up - bb_dn, 0.5);
   double bb_width = SnpDiv(bb_up - bb_dn, sma20[i]);

   //================= structure ===================================
   double dcup[], dcdn[];
   SnpRollingMax(h, n, 20, dcup);
   SnpRollingMin(l, n, 20, dcdn);
   double dist_to_res  = SnpDiv(dcup[i] - c[i], c[i]);
   double dist_to_sup  = SnpDiv(c[i] - dcdn[i], c[i]);
   double donch_width  = SnpDiv(dcup[i] - dcdn[i], c[i]);
   double close_z      = SnpDiv(c[i] - sma20[i], sd20[i]);

   //================= candle ======================================
   double rng = h[i] - l[i];
   double range_pct      = SnpDiv(rng, c[i]);
   double body_pct       = SnpDiv(c[i] - o[i], rng);
   double upper_wick_pct = SnpDiv(h[i] - MathMax(o[i], c[i]), rng);
   double lower_wick_pct = SnpDiv(MathMin(o[i], c[i]) - l[i], rng);

   //================= order flow (TICK volume - see docs/REVIEW.md) =
   double volsma20[];
   SnpSMA(v, n, 20, volsma20);
   double vol_delta = SnpDiv(v[i] - v[i-1], v[i-1] + 1.0);
   double rvol_20   = SnpDiv(v[i], volsma20[i], 1.0);

   //================= stochastic ==================================
   double hh14[], ll14[], stk[], std3[];
   SnpRollingMax(h, n, 14, hh14);
   SnpRollingMin(l, n, 14, ll14);
   ArrayResize(stk, n);
   for(int k = 0; k < n; k++)
      stk[k] = 100.0 * SnpDiv(c[k] - ll14[k], hh14[k] - ll14[k], 0.5);
   SnpSMA(stk, n, 3, std3);

   //================= time ========================================
   //--- fractional hour so an M5 bar at 13:35 differs from 13:05.
   //--- day_of_week uses the MetaTrader convention: 0 = Sunday.
   MqlDateTime dt;
   TimeToStruct(t[i], dt);
   double frac_hour = dt.hour + dt.min / 60.0;
   double hour_sin = MathSin(2.0 * M_PI * frac_hour / 24.0);
   double hour_cos = MathCos(2.0 * M_PI * frac_hour / 24.0);
   double dow_sin  = MathSin(2.0 * M_PI * dt.day_of_week / 7.0);
   double dow_cos  = MathCos(2.0 * M_PI * dt.day_of_week / 7.0);

   //================= assemble ====================================
   out[F_RET_1]            = (float)f_ret1;
   out[F_RET_3]            = (float)f_ret3;
   out[F_RET_5]            = (float)f_ret5;
   out[F_RET_10]           = (float)f_ret10;
   out[F_RET_1_LAG1]       = (float)f_ret1_l1;
   out[F_RET_1_LAG2]       = (float)f_ret1_l2;
   out[F_RET_1_LAG3]       = (float)f_ret1_l3;
   out[F_RSI_7]            = (float)rsi7[i];
   out[F_RSI_14]           = (float)rsi14[i];
   out[F_RSI_14_LAG1]      = (float)rsi14[i-1];
   out[F_MACD]             = (float)macd_n;
   out[F_MACD_SIGNAL]      = (float)macdsig_n;
   out[F_MACD_HIST]        = (float)macdhist_n;
   out[F_MACD_HIST_LAG1]   = (float)macdhist_l1;
   out[F_ADX_14]           = (float)adx[i];
   out[F_PLUS_DI_14]       = (float)pdi[i];
   out[F_MINUS_DI_14]      = (float)mdi[i];
   out[F_EMA20_DIST]       = (float)ema20_dist;
   out[F_SMA60_DIST]       = (float)sma60_dist;
   out[F_EMA20_SLOPE]      = (float)ema20_slope;
   out[F_ATR14_NORM]       = (float)atr14_norm;
   out[F_ATR_RATIO]        = (float)atr_ratio;
   out[F_BB_PCTB]          = (float)bb_pctb;
   out[F_BB_WIDTH]         = (float)bb_width;
   out[F_DIST_TO_RES]      = (float)dist_to_res;
   out[F_DIST_TO_SUP]      = (float)dist_to_sup;
   out[F_DONCHIAN_WIDTH]   = (float)donch_width;
   out[F_CLOSE_ZSCORE_20]  = (float)close_z;
   out[F_RANGE_PCT]        = (float)range_pct;
   out[F_BODY_PCT]         = (float)body_pct;
   out[F_UPPER_WICK_PCT]   = (float)upper_wick_pct;
   out[F_LOWER_WICK_PCT]   = (float)lower_wick_pct;
   out[F_VOL_DELTA]        = (float)vol_delta;
   out[F_RVOL_20]          = (float)rvol_20;
   out[F_STOCH_K_14]       = (float)stk[i];
   out[F_STOCH_D_3]        = (float)std3[i];
   out[F_HOUR_SIN]         = (float)hour_sin;
   out[F_HOUR_COS]         = (float)hour_cos;
   out[F_DOW_SIN]          = (float)dow_sin;
   out[F_DOW_COS]          = (float)dow_cos;

   //--- final NaN/Inf sweep. One bad float and ONNX returns nonsense
   //--- probabilities without ever raising an error.
   for(int k = 0; k < SNP_FEATURE_DIM; k++)
     {
      if(!MathIsValidNumber(out[k]))
        {
         m_error = StringFormat("feature %s is not finite", SnpFeatureName(k));
         return(false);
        }
     }

   m_last_atr   = atr14[i];
   m_last_adx   = adx[i];
   m_last_rsi14 = rsi14[i];
   return(true);
  }

//+------------------------------------------------------------------+
//| Meta feature vector: the 40 base features plus what the primary  |
//| model just said about them. This is the Lopez de Prado           |
//| meta-labelling input - the second model answers "given this      |
//| setup AND this call, do I take the bet?", never "which way?".    |
//+------------------------------------------------------------------+
void CSnpFeatureEngine::BuildMeta(const float &base[], const double p_up, const double side, float &meta[])
  {
   ArrayResize(meta, SNP_META_DIM);
   for(int k = 0; k < SNP_FEATURE_DIM; k++)
      meta[k] = base[k];
   meta[SNP_FEATURE_DIM + 0] = (float)p_up;
   meta[SNP_FEATURE_DIM + 1] = (float)side;
   meta[SNP_FEATURE_DIM + 2] = (float)MathAbs(2.0 * p_up - 1.0);
  }

#endif // __SNIPER_FEATURES_MQH__
