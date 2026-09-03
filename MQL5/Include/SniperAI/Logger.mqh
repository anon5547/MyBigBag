//+------------------------------------------------------------------+
//|                                                       Logger.mqh |
//|  SniperAI V5 - journal + on-chart ring buffer + CSV audit trail. |
//|                                                                  |
//|  The CSV is not decoration. When a live run disagrees with your  |
//|  backtest you will need the exact feature vector and probability |
//|  that produced each decision, and you will need it months later. |
//+------------------------------------------------------------------+
#property copyright "SniperAI V5"
#ifndef __SNIPER_LOGGER_MQH__
#define __SNIPER_LOGGER_MQH__

#define SNP_LOG_RING 40

enum ENUM_SNP_LOGLEVEL
  {
   SNP_LOG_ERROR = 0,
   SNP_LOG_WARN  = 1,
   SNP_LOG_INFO  = 2,
   SNP_LOG_DEBUG = 3
  };

class CSnpLogger
  {
private:
   string            m_ring[SNP_LOG_RING];
   int               m_count;
   int               m_head;
   ENUM_SNP_LOGLEVEL m_level;
   bool              m_to_journal;
   string            m_csv_path;
   int               m_csv_handle;
   bool              m_dirty;

   void              Push(const string line);

public:
                     CSnpLogger();
                    ~CSnpLogger();

   void              Configure(const ENUM_SNP_LOGLEVEL lvl, const bool journal, const string csv_path);
   void              Close();

   void              Error(const string tag, const string msg) { Write(SNP_LOG_ERROR, tag, msg); }
   void              Warn (const string tag, const string msg) { Write(SNP_LOG_WARN,  tag, msg); }
   void              Info (const string tag, const string msg) { Write(SNP_LOG_INFO,  tag, msg); }
   void              Debug(const string tag, const string msg) { Write(SNP_LOG_DEBUG, tag, msg); }
   void              Write(const ENUM_SNP_LOGLEVEL lvl, const string tag, const string msg);

   //--- structured decision record, one row per closed bar
   void              Decision(const datetime bar_time, const string sym,
                              const double p_up, const double meta_conf,
                              const string action, const string detail,
                              const float &features[]);

   int               Count() const { return(m_count); }
   string            Line(const int i) const;   // i = 0 -> oldest kept line
   bool              Dirty() const { return(m_dirty); }
   void              ClearDirty()  { m_dirty = false; }
   void              Clear();
  };

//+------------------------------------------------------------------+
CSnpLogger::CSnpLogger()
  {
   m_count      = 0;
   m_head       = 0;
   m_level      = SNP_LOG_INFO;
   m_to_journal = true;
   m_csv_path   = "";
   m_csv_handle = INVALID_HANDLE;
   m_dirty      = true;
   for(int i = 0; i < SNP_LOG_RING; i++) m_ring[i] = "";
  }

CSnpLogger::~CSnpLogger() { Close(); }

//+------------------------------------------------------------------+
void CSnpLogger::Configure(const ENUM_SNP_LOGLEVEL lvl, const bool journal, const string csv_path)
  {
   m_level      = lvl;
   m_to_journal = journal;
   Close();
   m_csv_path = csv_path;
   if(m_csv_path == "")
      return;

   bool fresh = !FileIsExist(m_csv_path);
   //--- FILE_TXT, not FILE_CSV: we build the row ourselves so no
   //--- delimiter re-interpretation can ever corrupt a feature column.
   m_csv_handle = FileOpen(m_csv_path, FILE_WRITE | FILE_READ | FILE_TXT | FILE_ANSI);
   if(m_csv_handle == INVALID_HANDLE)
     {
      Write(SNP_LOG_WARN, "LOG", StringFormat("เปิดไฟล์ CSV ไม่ได้ (%s) err=%d", m_csv_path, GetLastError()));
      return;
     }
   FileSeek(m_csv_handle, 0, SEEK_END);
   if(fresh)
     {
      string hdr = "utc_time,bar_time,symbol,p_up,meta_conf,action,detail";
      for(int i = 0; i < 40; i++)
         hdr += ",f" + IntegerToString(i);
      FileWriteString(m_csv_handle, hdr + "\r\n");
      FileFlush(m_csv_handle);
     }
  }

//+------------------------------------------------------------------+
void CSnpLogger::Close()
  {
   if(m_csv_handle != INVALID_HANDLE)
     {
      FileFlush(m_csv_handle);
      FileClose(m_csv_handle);
      m_csv_handle = INVALID_HANDLE;
     }
  }

//+------------------------------------------------------------------+
void CSnpLogger::Push(const string line)
  {
   m_ring[m_head] = line;
   m_head = (m_head + 1) % SNP_LOG_RING;
   if(m_count < SNP_LOG_RING) m_count++;
   m_dirty = true;
  }

//+------------------------------------------------------------------+
string CSnpLogger::Line(const int i) const
  {
   if(i < 0 || i >= m_count) return("");
   int start = (m_count < SNP_LOG_RING) ? 0 : m_head;
   return(m_ring[(start + i) % SNP_LOG_RING]);
  }

//+------------------------------------------------------------------+
void CSnpLogger::Clear()
  {
   m_count = 0;
   m_head  = 0;
   for(int i = 0; i < SNP_LOG_RING; i++) m_ring[i] = "";
   m_dirty = true;
  }

//+------------------------------------------------------------------+
void CSnpLogger::Write(const ENUM_SNP_LOGLEVEL lvl, const string tag, const string msg)
  {
   if(lvl > m_level) return;
   string stamp = TimeToString(TimeCurrent(), TIME_MINUTES | TIME_SECONDS);
   string line  = StringFormat("[%s] [%s] %s", stamp, tag, msg);
   Push(line);
   if(m_to_journal)
      Print(line);
  }

//+------------------------------------------------------------------+
void CSnpLogger::Decision(const datetime bar_time, const string sym,
                          const double p_up, const double meta_conf,
                          const string action, const string detail,
                          const float &features[])
  {
   if(m_csv_handle == INVALID_HANDLE) return;
   string row = StringFormat("%s,%s,%s,%.6f,%.6f,%s,%s",
                             TimeToString(TimeGMT(), TIME_DATE | TIME_SECONDS),
                             TimeToString(bar_time,  TIME_DATE | TIME_SECONDS),
                             sym, p_up, meta_conf, action, detail);
   int n = ArraySize(features);
   for(int i = 0; i < n; i++)
      row += StringFormat(",%.8g", features[i]);
   FileWriteString(m_csv_handle, row + "\r\n");
   FileFlush(m_csv_handle);
  }

#endif // __SNIPER_LOGGER_MQH__
