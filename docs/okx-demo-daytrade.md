# OKX 模擬盤當沖：開盤 1 小時區間突破

這支程式只連 **OKX 模擬交易**（假錢），沒有切到真錢的選項。目的是讓你用假錢親眼看到
這套當沖規則的實際結果。

## 規則

以 UTC 日為一天（台灣時間每天 08:00 換日）：

1. 台灣 08:00–09:00 這 1 小時的最高價、最低價，是今天的「開盤區間」。
2. 之後任何一根 15 分鐘 K 線收盤高於區間最高價，就用固定金額市價買進。每天最多買一次。
3. 價格跌到區間最低價就停損賣出。
4. 台灣隔天 07:45 一定全部賣出，不留倉過夜。

BTC 2018–2026 的回測：不算手續費時年化約 +42%，扣掉 OKX 一般手續費（單邊約 0.1%，
加滑價 0.15%）後年化約 **-40%**。手續費是當沖最大的成本。

## 第一次設定

1. 登入 OKX 網站或 App，切換到「模擬交易」，在模擬交易裡建立 API key
   （權限只勾「交易」，不要勾「提領」）。正式帳戶的 key 不能用在模擬盤。
2. 在電腦安裝 Python 3.10 以上，下載這個專案後在專案資料夾執行：

   ```bash
   pip install -e ".[okx]"
   ```

3. 設定三個環境變數（Windows 命令提示字元）：

   ```bat
   set OKX_API_KEY=你的模擬盤 API Key
   set OKX_API_SECRET=你的模擬盤 Secret
   set OKX_API_PASSPHRASE=你建立 key 時設定的密碼
   ```

   macOS / Linux 把 `set` 換成 `export`。

## 執行

```bash
python -m core.live.okx_orb --symbol BTC/USDT --budget 100
```

- `--budget`：每次進場用多少 USDT（模擬盤的假錢）。
- 程式每 30 秒檢查一次，視窗要一直開著；電腦睡眠或關機時不會交易。
- 按 `Ctrl+C` 停止。狀態存在 `okx_orb_state.json`，重開會接續，不會重複買。
- 每筆交易記錄在 `okx_orb_trades.csv`，跑幾週後可以拿去給反詐投資王分析：

  ```bash
  anti-gambling-trader analyze okx_orb_trades.csv
  ```
