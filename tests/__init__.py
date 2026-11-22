# 1..25 | ForEach-Object {
#     $body = @{ message = "Hammer test $_"; thread_id = "test" } | ConvertTo-Json
#     try {
#         $status = (Invoke-WebRequest -Uri "http://localhost:8000/chat" -Method Post -Body $body -ContentType "application/json" -UseBasicParsing).StatusCode
#         Write-Host "Request $_`: HTTP $status" -ForegroundColor Green
#     } catch {
#         $status = $_.Exception.Response.StatusCode.value__
#         Write-Host "Request $_`: HTTP $status" -ForegroundColor Red
#     }
# }