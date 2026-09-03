//+------------------------------------------------------------------+
//|                                                        Panel.mqh |
//|  SniperAI V5 - on-chart control panel.                           |
//|                                                                  |
//|  This is the tkinter "Sniper AI Control Panel V4" rebuilt as     |
//|  native chart objects: same fields, same buttons, same green-on- |
//|  black log - but it lives inside the terminal, so it cannot      |
//|  freeze the trading loop and it cannot lose the connection       |
//|  separately from MT5.                                            |
//+------------------------------------------------------------------+
#property copyright "SniperAI V5"
#ifndef __SNIPER_PANEL_MQH__
#define __SNIPER_PANEL_MQH__

#include <SniperAI/Logger.mqh>

#define SNP_P               "SniperV5_"
#define SNP_PANEL_W         430
#define SNP_LOG_LINES       14
#define SNP_ROW_H           22

//--- which editable field changed
enum ENUM_SNP_EDIT
  {
   SNP_EDIT_NONE = 0,
   SNP_EDIT_CONF,
   SNP_EDIT_LOT,
   SNP_EDIT_TP,
   SNP_EDIT_SL,
   SNP_EDIT_TRAIL_START,
   SNP_EDIT_TRAIL_DIST
  };

//+------------------------------------------------------------------+
class CSnpPanel
  {
private:
   long              m_chart;
   int               m_x, m_y;
   bool              m_built;
   bool              m_running;

   //--- editable values, owned by the panel between syncs
   double            m_conf, m_lot, m_tp, m_sl, m_trail_start, m_trail_dist;

   void              Label(const string id, const int x, const int y, const string text,
                           const color clr, const int size = 8, const string font = "Tahoma");
   void              Rect(const string id, const int x, const int y, const int w, const int h,
                          const color bg, const color border);
   void              Edit(const string id, const int x, const int y, const int w,
                          const string text);
   void              Button(const string id, const int x, const int y, const int w, const int h,
                            const string text, const color bg, const color fg);
   void              SetText(const string id, const string text);
   void              SetColor(const string id, const color clr);

public:
                     CSnpPanel();
   void              Create(const long chart_id, const int x, const int y);
   void              Destroy();
   bool              IsBuilt()  const { return(m_built);   }
   bool              IsRunning()const { return(m_running); }
   void              SetRunning(const bool on);

   //--- push EA state into the panel
   void              SyncInputs(const double conf, const double lot, const double tp,
                                const double sl, const double tstart, const double tdist);
   void              SetStatus(const string key, const string value, const color clr = clrWhite);
   void              SetHeadline(const string text, const color clr);
   void              RenderLog(CSnpLogger &logger);

   //--- read back what the user typed
   double            Conf()       const { return(m_conf);        }
   double            Lot()        const { return(m_lot);         }
   double            TpMoney()    const { return(m_tp);          }
   double            SlMoney()    const { return(m_sl);          }
   double            TrailStart() const { return(m_trail_start); }
   double            TrailDist()  const { return(m_trail_dist);  }

   //--- returns which field the user just committed (SNP_EDIT_NONE if not ours)
   ENUM_SNP_EDIT     OnEndEdit(const string sparam);
   //--- returns 1 = start pressed, -1 = stop pressed, 0 = not ours
   int               OnClick(const string sparam);
  };

//+------------------------------------------------------------------+
CSnpPanel::CSnpPanel()
  {
   m_chart = 0; m_x = 12; m_y = 24;
   m_built = false; m_running = false;
   m_conf = 0.35; m_lot = 0.05; m_tp = 100.0; m_sl = 10.0;
   m_trail_start = 1.0; m_trail_dist = 0.5;
  }

//+------------------------------------------------------------------+
void CSnpPanel::Label(const string id, const int x, const int y, const string text,
                      const color clr, const int size, const string font)
  {
   string n = SNP_P + id;
   if(ObjectFind(m_chart, n) < 0)
      ObjectCreate(m_chart, n, OBJ_LABEL, 0, 0, 0);
   ObjectSetInteger(m_chart, n, OBJPROP_CORNER, CORNER_LEFT_UPPER);
   ObjectSetInteger(m_chart, n, OBJPROP_XDISTANCE, x);
   ObjectSetInteger(m_chart, n, OBJPROP_YDISTANCE, y);
   ObjectSetInteger(m_chart, n, OBJPROP_COLOR, clr);
   ObjectSetInteger(m_chart, n, OBJPROP_FONTSIZE, size);
   ObjectSetInteger(m_chart, n, OBJPROP_SELECTABLE, false);
   ObjectSetInteger(m_chart, n, OBJPROP_HIDDEN, true);
   ObjectSetInteger(m_chart, n, OBJPROP_BACK, false);
   ObjectSetString (m_chart, n, OBJPROP_FONT, font);
   ObjectSetString (m_chart, n, OBJPROP_TEXT, text);
  }

//+------------------------------------------------------------------+
void CSnpPanel::Rect(const string id, const int x, const int y, const int w, const int h,
                     const color bg, const color border)
  {
   string n = SNP_P + id;
   if(ObjectFind(m_chart, n) < 0)
      ObjectCreate(m_chart, n, OBJ_RECTANGLE_LABEL, 0, 0, 0);
   ObjectSetInteger(m_chart, n, OBJPROP_CORNER, CORNER_LEFT_UPPER);
   ObjectSetInteger(m_chart, n, OBJPROP_XDISTANCE, x);
   ObjectSetInteger(m_chart, n, OBJPROP_YDISTANCE, y);
   ObjectSetInteger(m_chart, n, OBJPROP_XSIZE, w);
   ObjectSetInteger(m_chart, n, OBJPROP_YSIZE, h);
   ObjectSetInteger(m_chart, n, OBJPROP_BGCOLOR, bg);
   ObjectSetInteger(m_chart, n, OBJPROP_BORDER_TYPE, BORDER_FLAT);
   ObjectSetInteger(m_chart, n, OBJPROP_COLOR, border);
   ObjectSetInteger(m_chart, n, OBJPROP_SELECTABLE, false);
   ObjectSetInteger(m_chart, n, OBJPROP_HIDDEN, true);
   ObjectSetInteger(m_chart, n, OBJPROP_BACK, false);
  }

//+------------------------------------------------------------------+
void CSnpPanel::Edit(const string id, const int x, const int y, const int w, const string text)
  {
   string n = SNP_P + id;
   if(ObjectFind(m_chart, n) < 0)
      ObjectCreate(m_chart, n, OBJ_EDIT, 0, 0, 0);
   ObjectSetInteger(m_chart, n, OBJPROP_CORNER, CORNER_LEFT_UPPER);
   ObjectSetInteger(m_chart, n, OBJPROP_XDISTANCE, x);
   ObjectSetInteger(m_chart, n, OBJPROP_YDISTANCE, y);
   ObjectSetInteger(m_chart, n, OBJPROP_XSIZE, w);
   ObjectSetInteger(m_chart, n, OBJPROP_YSIZE, 18);
   ObjectSetInteger(m_chart, n, OBJPROP_BGCOLOR, clrWhite);
   ObjectSetInteger(m_chart, n, OBJPROP_COLOR, clrBlack);
   ObjectSetInteger(m_chart, n, OBJPROP_BORDER_COLOR, clrGray);
   ObjectSetInteger(m_chart, n, OBJPROP_ALIGN, ALIGN_LEFT);
   ObjectSetInteger(m_chart, n, OBJPROP_FONTSIZE, 8);
   ObjectSetInteger(m_chart, n, OBJPROP_SELECTABLE, false);
   ObjectSetInteger(m_chart, n, OBJPROP_HIDDEN, true);
   ObjectSetString (m_chart, n, OBJPROP_FONT, "Tahoma");
   ObjectSetString (m_chart, n, OBJPROP_TEXT, text);
  }

//+------------------------------------------------------------------+
void CSnpPanel::Button(const string id, const int x, const int y, const int w, const int h,
                       const string text, const color bg, const color fg)
  {
   string n = SNP_P + id;
   if(ObjectFind(m_chart, n) < 0)
      ObjectCreate(m_chart, n, OBJ_BUTTON, 0, 0, 0);
   ObjectSetInteger(m_chart, n, OBJPROP_CORNER, CORNER_LEFT_UPPER);
   ObjectSetInteger(m_chart, n, OBJPROP_XDISTANCE, x);
   ObjectSetInteger(m_chart, n, OBJPROP_YDISTANCE, y);
   ObjectSetInteger(m_chart, n, OBJPROP_XSIZE, w);
   ObjectSetInteger(m_chart, n, OBJPROP_YSIZE, h);
   ObjectSetInteger(m_chart, n, OBJPROP_BGCOLOR, bg);
   ObjectSetInteger(m_chart, n, OBJPROP_COLOR, fg);
   ObjectSetInteger(m_chart, n, OBJPROP_BORDER_COLOR, clrDimGray);
   ObjectSetInteger(m_chart, n, OBJPROP_FONTSIZE, 9);
   ObjectSetInteger(m_chart, n, OBJPROP_STATE, false);
   ObjectSetInteger(m_chart, n, OBJPROP_SELECTABLE, false);
   ObjectSetInteger(m_chart, n, OBJPROP_HIDDEN, true);
   ObjectSetString (m_chart, n, OBJPROP_FONT, "Tahoma");
   ObjectSetString (m_chart, n, OBJPROP_TEXT, text);
  }

//+------------------------------------------------------------------+
void CSnpPanel::SetText(const string id, const string text)
  {
   string n = SNP_P + id;
   if(ObjectFind(m_chart, n) >= 0)
      ObjectSetString(m_chart, n, OBJPROP_TEXT, text);
  }

void CSnpPanel::SetColor(const string id, const color clr)
  {
   string n = SNP_P + id;
   if(ObjectFind(m_chart, n) >= 0)
      ObjectSetInteger(m_chart, n, OBJPROP_COLOR, clr);
  }

//+------------------------------------------------------------------+
void CSnpPanel::Create(const long chart_id, const int x, const int y)
  {
   m_chart = chart_id;
   m_x = x;
   m_y = y;

   int total_h = 8 * SNP_ROW_H + 34 + 8 * 16 + SNP_LOG_LINES * 13 + 40;
   Rect("bg", m_x - 6, m_y - 6, SNP_PANEL_W, total_h, C'26,28,34', C'70,80,95');

   int yy = m_y;
   Label("title", m_x + 6, yy, "Sniper AI Control Panel V5  ·  MQL5 native", C'120,200,255', 10);
   yy += 20;
   Label("subtitle", m_x + 6, yy, "two-tier meta-labelling · ONNX · server-side stops", C'130,140,155', 7);
   yy += 20;

   int lx = m_x + 6;         // label column
   int ex = m_x + 250;       // edit column
   int ew = 90;

   Label("l_conf",  lx, yy + 3, "AI Confidence Threshold (0-1):", clrGainsboro);
   Edit ("e_conf",  ex, yy, ew, DoubleToString(m_conf, 2));
   yy += SNP_ROW_H;

   Label("l_lot",   lx, yy + 3, "Lot Size (Volume):", clrGainsboro);
   Edit ("e_lot",   ex, yy, ew, DoubleToString(m_lot, 2));
   yy += SNP_ROW_H;

   Label("l_tp",    lx, yy + 3, "TP Money ($):", clrGainsboro);
   Edit ("e_tp",    ex, yy, ew, DoubleToString(m_tp, 2));
   yy += SNP_ROW_H;

   Label("l_sl",    lx, yy + 3, "SL Money ($):", clrGainsboro);
   Edit ("e_sl",    ex, yy, ew, DoubleToString(m_sl, 2));
   yy += SNP_ROW_H;

   Label("l_ts",    lx, yy + 3, "Trailing Start ($):", clrGainsboro);
   Edit ("e_ts",    ex, yy, ew, DoubleToString(m_trail_start, 2));
   yy += SNP_ROW_H;

   Label("l_td",    lx, yy + 3, "Trailing Distance ($):", clrGainsboro);
   Edit ("e_td",    ex, yy, ew, DoubleToString(m_trail_dist, 2));
   yy += SNP_ROW_H + 6;

   Button("b_start", lx,       yy, 150, 26, "Start Bot", C'22,110,60',  clrWhite);
   Button("b_stop",  lx + 180, yy, 150, 26, "Stop Bot",  C'130,40,45',  clrWhite);
   yy += 34;

   Label("h_state", lx, yy, "STOPPED", clrOrangeRed, 10);
   yy += 18;

   //--- status block ------------------------------------------------
   string keys[]  = {"model", "signal", "prob", "meta", "pos", "pnl", "spread", "gate"};
   string names[] = {"Model", "Signal", "P(up)/P(dn)", "Meta conf", "Positions",
                     "Day P/L", "Spread", "Gate"};
   for(int i = 0; i < ArraySize(keys); i++)
     {
      Label("k_" + keys[i], lx,       yy, names[i] + ":", C'120,130,145', 7);
      Label("v_" + keys[i], lx + 110, yy, "-",            clrWhite,       7);
      yy += 16;
     }

   //--- log block ---------------------------------------------------
   yy += 4;
   Rect("logbg", lx - 2, yy - 2, SNP_PANEL_W - 14, SNP_LOG_LINES * 13 + 8, C'8,10,12', C'50,60,70');
   for(int i = 0; i < SNP_LOG_LINES; i++)
      Label("log" + IntegerToString(i), lx + 2, yy + 2 + i * 13, "", C'60,220,90', 7, "Consolas");

   m_built = true;
   ChartRedraw(m_chart);
  }

//+------------------------------------------------------------------+
void CSnpPanel::Destroy()
  {
   ObjectsDeleteAll(m_chart, SNP_P);
   m_built = false;
   ChartRedraw(m_chart);
  }

//+------------------------------------------------------------------+
void CSnpPanel::SetRunning(const bool on)
  {
   m_running = on;
   if(!m_built) return;
   SetText ("h_state", on ? "RUNNING" : "STOPPED");
   SetColor("h_state", on ? C'60,220,90' : clrOrangeRed);
   ObjectSetInteger(m_chart, SNP_P + "b_start", OBJPROP_BGCOLOR, on ? C'40,60,50' : C'22,110,60');
   ObjectSetInteger(m_chart, SNP_P + "b_stop",  OBJPROP_BGCOLOR, on ? C'130,40,45' : C'60,40,42');
   ChartRedraw(m_chart);
  }

//+------------------------------------------------------------------+
void CSnpPanel::SyncInputs(const double conf, const double lot, const double tp,
                           const double sl, const double tstart, const double tdist)
  {
   m_conf = conf; m_lot = lot; m_tp = tp; m_sl = sl;
   m_trail_start = tstart; m_trail_dist = tdist;
   if(!m_built) return;
   SetText("e_conf", DoubleToString(m_conf, 2));
   SetText("e_lot",  DoubleToString(m_lot, 2));
   SetText("e_tp",   DoubleToString(m_tp, 2));
   SetText("e_sl",   DoubleToString(m_sl, 2));
   SetText("e_ts",   DoubleToString(m_trail_start, 2));
   SetText("e_td",   DoubleToString(m_trail_dist, 2));
  }

//+------------------------------------------------------------------+
void CSnpPanel::SetStatus(const string key, const string value, const color clr)
  {
   if(!m_built) return;
   SetText ("v_" + key, value);
   SetColor("v_" + key, clr);
  }

void CSnpPanel::SetHeadline(const string text, const color clr)
  {
   if(!m_built) return;
   SetText ("h_state", text);
   SetColor("h_state", clr);
  }

//+------------------------------------------------------------------+
void CSnpPanel::RenderLog(CSnpLogger &logger)
  {
   if(!m_built) return;
   int total = logger.Count();
   int first = MathMax(0, total - SNP_LOG_LINES);
   for(int i = 0; i < SNP_LOG_LINES; i++)
     {
      int src = first + i;
      string txt = (src < total) ? logger.Line(src) : "";
      if(StringLen(txt) > 68)
         txt = StringSubstr(txt, 0, 68);
      SetText("log" + IntegerToString(i), txt);
     }
   ChartRedraw(m_chart);
  }

//+------------------------------------------------------------------+
ENUM_SNP_EDIT CSnpPanel::OnEndEdit(const string sparam)
  {
   if(!m_built) return(SNP_EDIT_NONE);
   if(StringFind(sparam, SNP_P) != 0) return(SNP_EDIT_NONE);

   string id  = StringSubstr(sparam, StringLen(SNP_P));
   string raw = ObjectGetString(m_chart, sparam, OBJPROP_TEXT);
   StringReplace(raw, ",", ".");                 // Thai keyboards, decimal comma
   double v   = StringToDouble(raw);

   if(id == "e_conf")
     {
      if(v < 0.0) v = 0.0;
      if(v > 1.0) v = 1.0;
      m_conf = v;
      SetText("e_conf", DoubleToString(m_conf, 2));
      return(SNP_EDIT_CONF);
     }
   if(id == "e_lot")
     {
      if(v < 0.0) v = 0.0;
      m_lot = v;
      SetText("e_lot", DoubleToString(m_lot, 2));
      return(SNP_EDIT_LOT);
     }
   if(id == "e_tp")
     {
      m_tp = MathMax(0.0, v);
      SetText("e_tp", DoubleToString(m_tp, 2));
      return(SNP_EDIT_TP);
     }
   if(id == "e_sl")
     {
      m_sl = MathMax(0.0, v);
      SetText("e_sl", DoubleToString(m_sl, 2));
      return(SNP_EDIT_SL);
     }
   if(id == "e_ts")
     {
      m_trail_start = MathMax(0.0, v);
      SetText("e_ts", DoubleToString(m_trail_start, 2));
      return(SNP_EDIT_TRAIL_START);
     }
   if(id == "e_td")
     {
      m_trail_dist = MathMax(0.0, v);
      SetText("e_td", DoubleToString(m_trail_dist, 2));
      return(SNP_EDIT_TRAIL_DIST);
     }
   return(SNP_EDIT_NONE);
  }

//+------------------------------------------------------------------+
int CSnpPanel::OnClick(const string sparam)
  {
   if(!m_built) return(0);
   if(sparam == SNP_P + "b_start")
     {
      ObjectSetInteger(m_chart, sparam, OBJPROP_STATE, false);
      return(1);
     }
   if(sparam == SNP_P + "b_stop")
     {
      ObjectSetInteger(m_chart, sparam, OBJPROP_STATE, false);
      return(-1);
     }
   return(0);
  }

#endif // __SNIPER_PANEL_MQH__
