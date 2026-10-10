# 安全政策

## 回報安全問題

若你發現本專案的安全漏洞(例如:產生的 HTML 有 XSS、scaffold 產出的
專案會洩漏金鑰、路徑穿越),請透過 GitHub 的
[Private vulnerability reporting](https://github.com/mars-tw/anti-gambling-trader-tw/security/advisories/new)
回報,不要開公開 Issue。

## 設計上的安全邊界

- 本工具**不連網**:所有分析在本機執行,不上傳任何交易資料。
- 本體**不接收、不上傳任何金鑰**。scaffold 產出的專案會引導使用者把
  credentials 填在**本機** `config.yaml`(預設被 `.gitignore` 排除,
  不會進版控);金鑰的保管責任在使用者本機環境。
- 真實下單有**雙重閘門**(`ALLOW_LIVE_TRADING` 常數 + config 確認),
  預設全部封鎖;繞過閘門的風險由使用者自負。
