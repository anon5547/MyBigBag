//+------------------------------------------------------------------+
//|                                                    Inference.mqh |
//|  SniperAI V5 - two-tier meta-labelling inference, native ONNX.   |
//|                                                                  |
//|  TIER 1 (primary)  : 40 features -> P(up)                        |
//|  TIER 2 (meta)     : 43 features -> P(take the trade)            |
//|                                                                  |
//|  The meta model is a GATE, not a second opinion on direction.    |
//|  If you ever find yourself using its output to flip the side,    |
//|  you have rebuilt a single-tier model with extra steps.          |
//|                                                                  |
//|  Requires MetaTrader 5 build 3620+ (native ONNX runtime).        |
//+------------------------------------------------------------------+
#property copyright "SniperAI V5"
#ifndef __SNIPER_INFERENCE_MQH__
#define __SNIPER_INFERENCE_MQH__

#include <SniperAI/Features.mqh>

enum ENUM_SNP_SIDE { SNP_SIDE_NONE = 0, SNP_SIDE_BUY = 1, SNP_SIDE_SELL = -1 };

struct SnpPrediction
  {
   ENUM_SNP_SIDE     side;          // proposed direction
   double            p_up;          // primary P(up)
   double            p_down;        // 1 - p_up
   double            meta_conf;     // P(take) from the gate
   double            edge;          // |2*p_up-1|
   bool              valid;
   string            reason;        // why it is invalid / refused
  };

void SnpResetPrediction(SnpPrediction &p)
  {
   p.side      = SNP_SIDE_NONE;
   p.p_up      = 0.5;
   p.p_down    = 0.5;
   p.meta_conf = 0.0;
   p.edge      = 0.0;
   p.valid     = false;
   p.reason    = "";
  }

//+------------------------------------------------------------------+
//| CSnpInference                                                    |
//+------------------------------------------------------------------+
class CSnpInference
  {
private:
   long              m_h_primary;
   long              m_h_meta;
   bool              m_ready;
   bool              m_fallback;          // heuristic mode, NOT the AI
   int               m_up_index;          // column of P(up) in the primary output
   bool              m_single_output;     // model emits probabilities only
   string            m_last_error;

   bool              RunBinary(const long handle, const float &input[], const int dim,
                               double &p0, double &p1);
   double            HeuristicPUp(const float &f[]);
   double            HeuristicMeta(const float &f[], const double p_up);

public:
                     CSnpInference();
                    ~CSnpInference();

   bool              Load(const string primary_file, const string meta_file,
                          const int up_index, const bool single_output);
   void              Release();
   void              EnableFallback(const bool on) { m_fallback = on; }

   bool              IsReady()   const { return(m_ready || m_fallback); }
   bool              IsFallback()const { return(m_fallback && !m_ready); }
   string            LastError() const { return(m_last_error); }

   bool              Predict(const float &features[], SnpPrediction &out);
  };

//+------------------------------------------------------------------+
CSnpInference::CSnpInference()
  {
   m_h_primary    = INVALID_HANDLE;
   m_h_meta       = INVALID_HANDLE;
   m_ready        = false;
   m_fallback     = false;
   m_up_index     = 1;
   m_single_output= false;
   m_last_error   = "";
  }

CSnpInference::~CSnpInference() { Release(); }

//+------------------------------------------------------------------+
void CSnpInference::Release()
  {
   if(m_h_primary != INVALID_HANDLE) { OnnxRelease(m_h_primary); m_h_primary = INVALID_HANDLE; }
   if(m_h_meta    != INVALID_HANDLE) { OnnxRelease(m_h_meta);    m_h_meta    = INVALID_HANDLE; }
   m_ready = false;
  }

//+------------------------------------------------------------------+
//| Load both tiers. Paths are relative to <terminal>\MQL5\Files.    |
//+------------------------------------------------------------------+
bool CSnpInference::Load(const string primary_file, const string meta_file,
                         const int up_index, const bool single_output)
  {
   Release();
   m_up_index      = (up_index == 0) ? 0 : 1;
   m_single_output = single_output;
   m_last_error    = "";

   if((int)TerminalInfoInteger(TERMINAL_BUILD) < 3620)
     {
      m_last_error = "MetaTrader build < 3620 has no ONNX runtime. Update the terminal.";
      return(false);
     }

   if(!FileIsExist(primary_file))
     {
      m_last_error = "primary model not found: MQL5\\Files\\" + primary_file;
      return(false);
     }
   if(!FileIsExist(meta_file))
     {
      m_last_error = "meta model not found: MQL5\\Files\\" + meta_file;
      return(false);
     }

   m_h_primary = OnnxCreate(primary_file, ONNX_DEFAULT);
   if(m_h_primary == INVALID_HANDLE)
     {
      m_last_error = StringFormat("OnnxCreate(primary) failed, err=%d", GetLastError());
      return(false);
     }
   m_h_meta = OnnxCreate(meta_file, ONNX_DEFAULT);
   if(m_h_meta == INVALID_HANDLE)
     {
      m_last_error = StringFormat("OnnxCreate(meta) failed, err=%d", GetLastError());
      Release();
      return(false);
     }

   //--- Pin the dynamic batch dimension to 1.
   //--- OnnxSetInputShape / OnnxSetOutputShape take ULONG shape arrays.
   const ulong in_p[]  = {1, SNP_FEATURE_DIM};
   const ulong in_m[]  = {1, SNP_META_DIM};
   const ulong lab[]   = {1};
   const ulong prob[]  = {1, 2};

   bool ok = true;
   if(!OnnxSetInputShape(m_h_primary, 0, in_p)) ok = false;
   if(!OnnxSetInputShape(m_h_meta,    0, in_m)) ok = false;
   if(m_single_output)
     {
      if(!OnnxSetOutputShape(m_h_primary, 0, prob)) ok = false;
      if(!OnnxSetOutputShape(m_h_meta,    0, prob)) ok = false;
     }
   else
     {
      if(!OnnxSetOutputShape(m_h_primary, 0, lab))  ok = false;
      if(!OnnxSetOutputShape(m_h_primary, 1, prob)) ok = false;
      if(!OnnxSetOutputShape(m_h_meta,    0, lab))  ok = false;
      if(!OnnxSetOutputShape(m_h_meta,    1, prob)) ok = false;
     }
   if(!ok)
     {
      m_last_error = StringFormat("Onnx shape setup failed, err=%d. "
                                  "Export with zipmap=False (see tools/export_to_onnx.py).", GetLastError());
      Release();
      return(false);
     }

   m_ready    = true;
   m_fallback = false;
   return(true);
  }

//+------------------------------------------------------------------+
//| One binary classification pass.                                  |
//+------------------------------------------------------------------+
bool CSnpInference::RunBinary(const long handle, const float &input[], const int dim,
                              double &p0, double &p1)
  {
   p0 = 0.5; p1 = 0.5;
   if(handle == INVALID_HANDLE) return(false);
   if(ArraySize(input) != dim)  return(false);

   //--- ONNX wants a rank-2 tensor [1, dim]
   float x[];
   ArrayResize(x, dim);
   for(int i = 0; i < dim; i++) x[i] = input[i];

   float  proba[];
   ArrayResize(proba, 2);
   bool ok;

   if(m_single_output)
     {
      ok = OnnxRun(handle, ONNX_DEFAULT, x, proba);
     }
   else
     {
      long label[];
      ArrayResize(label, 1);
      ok = OnnxRun(handle, ONNX_DEFAULT, x, label, proba);
     }

   if(!ok)
     {
      m_last_error = StringFormat("OnnxRun failed, err=%d", GetLastError());
      return(false);
     }

   double a = (double)proba[0];
   double b = (double)proba[1];
   if(!MathIsValidNumber(a) || !MathIsValidNumber(b))
     {
      m_last_error = "model returned NaN probabilities";
      return(false);
     }
   //--- renormalise: some boosted-tree exports drift by 1e-6
   double s = a + b;
   if(s > SNP_EPS) { a /= s; b /= s; }
   p0 = a;
   p1 = b;
   return(true);
  }

//+------------------------------------------------------------------+
//| DEMO-ONLY heuristic. This is NOT your model. It exists so the    |
//| EA is testable before the .onnx files land, and it is disabled   |
//| by default. Never run it on a live account.                      |
//+------------------------------------------------------------------+
double CSnpInference::HeuristicPUp(const float &f[])
  {
   double z = 0.0;
   z += 0.030 * (f[F_RSI_14]      - 50.0);
   z += 0.020 * (f[F_STOCH_K_14]  - 50.0);
   z += 0.050 * (f[F_PLUS_DI_14]  - f[F_MINUS_DI_14]);
   z += 120.0 * f[F_EMA20_DIST];
   z +=  60.0 * f[F_SMA60_DIST];
   z +=   0.8 * f[F_MACD_HIST];
   z +=   0.4 * f[F_CLOSE_ZSCORE_20];
   z  = SnpClamp(z, -12.0, 12.0);
   return(1.0 / (1.0 + MathExp(-z)));
  }

double CSnpInference::HeuristicMeta(const float &f[], const double p_up)
  {
   double edge   = MathAbs(2.0 * p_up - 1.0);
   double trend  = SnpClamp(f[F_ADX_14] / 40.0, 0.0, 1.0);
   double calm   = SnpClamp(1.5 - f[F_ATR_RATIO], 0.0, 1.0);
   return(SnpClamp(0.55 * edge + 0.30 * trend + 0.15 * calm, 0.0, 1.0));
  }

//+------------------------------------------------------------------+
//| Predict. Returns false only on a hard failure; a legitimate      |
//| "no trade" comes back as valid=true with a low meta_conf.        |
//+------------------------------------------------------------------+
bool CSnpInference::Predict(const float &features[], SnpPrediction &out)
  {
   SnpResetPrediction(out);

   if(ArraySize(features) != SNP_FEATURE_DIM)
     {
      out.reason = "bad feature vector size";
      return(false);
     }

   double p_up;

   if(m_ready)
     {
      double p0, p1;
      if(!RunBinary(m_h_primary, features, SNP_FEATURE_DIM, p0, p1))
        {
         out.reason = "primary: " + m_last_error;
         return(false);
        }
      p_up = (m_up_index == 1) ? p1 : p0;
     }
   else if(m_fallback)
     {
      p_up = HeuristicPUp(features);
     }
   else
     {
      out.reason = "no model loaded";
      return(false);
     }

   out.p_up   = SnpClamp(p_up, 0.0, 1.0);
   out.p_down = 1.0 - out.p_up;
   out.edge   = MathAbs(2.0 * out.p_up - 1.0);
   out.side   = (out.p_up >= 0.5) ? SNP_SIDE_BUY : SNP_SIDE_SELL;

   //--- tier 2: the gate
   float meta[];
   CSnpFeatureEngine::BuildMeta(features, out.p_up, (double)out.side, meta);

   if(m_ready)
     {
      double m0, m1;
      if(!RunBinary(m_h_meta, meta, SNP_META_DIM, m0, m1))
        {
         out.reason = "meta: " + m_last_error;
         return(false);
        }
      out.meta_conf = (m_up_index == 1) ? m1 : m0;   // class 1 == "trade was profitable"
     }
   else
     {
      out.meta_conf = HeuristicMeta(features, out.p_up);
     }

   out.meta_conf = SnpClamp(out.meta_conf, 0.0, 1.0);
   out.valid     = true;
   return(true);
  }

#endif // __SNIPER_INFERENCE_MQH__
