#!/usr/bin/env python3
"""
email_notify.py - 分析完成 Email 通知
"""
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime

SMTP_HOST   = "smtp.gmail.com"
SMTP_PORT   = 587
SMTP_USER   = "zhen712@g.ncu.edu.tw"
SMTP_PASS   = "gnvy dmuy dupc pbqh".replace(" ", "")
SENDER_NAME = "DentalVis"


def send_analysis_done(to_email: str, user_name: str, analysis_type: str,
                       result_url: str = "http://140.115.51.163:40111"):
    """分析完成通知信"""
    type_label = "牙齒初始化" if analysis_type == "init" else "菌斑分析"
    now_str    = datetime.now().strftime("%Y/%m/%d %H:%M")

    subject = f"[DentalVis] {type_label}已完成"

    html = f"""
<!DOCTYPE html>
<html lang="zh-TW">
<head><meta charset="UTF-8"></head>
<body style="margin:0;padding:0;background:#f4f6ef;font-family:'DM Sans',Arial,sans-serif;">
  <table width="100%" cellpadding="0" cellspacing="0" style="background:#f4f6ef;padding:40px 0;">
    <tr><td align="center">
      <table width="560" cellpadding="0" cellspacing="0" style="background:#ffffff;border-radius:16px;overflow:hidden;box-shadow:0 4px 24px rgba(3,105,94,0.10);">

        <!-- Header -->
        <tr><td style="background:#03695e;padding:28px 32px;">
          <div style="font-size:22px;font-weight:700;color:#ffffff;letter-spacing:0.5px;">DentalVis</div>
          <div style="font-size:13px;color:rgba(255,255,255,0.75);margin-top:4px;">Dental Health Analysis</div>
        </td></tr>

        <!-- Body -->
        <tr><td style="padding:32px;">
          <p style="margin:0 0 8px;font-size:16px;color:#1a2420;">Hi {user_name}，</p>
          <p style="margin:0 0 24px;font-size:15px;color:#5a7068;line-height:1.7;">
            您的 <strong style="color:#03695e;">{type_label}</strong> 已於 {now_str} 完成，請點擊下方按鈕查看結果。
          </p>

          <!-- CTA Button -->
          <table cellpadding="0" cellspacing="0" style="margin:0 0 28px;">
            <tr><td style="background:#03695e;border-radius:40px;padding:14px 32px;">
              <a href="{result_url}" style="color:#ffffff;font-size:15px;font-weight:600;text-decoration:none;display:block;">
                查看分析結果 →
              </a>
            </td></tr>
          </table>

          <!-- Info Box -->
          <table width="100%" cellpadding="0" cellspacing="0"
                 style="background:#f4f6ef;border-radius:10px;padding:16px 20px;margin-bottom:24px;">
            <tr>
              <td style="font-size:13px;color:#5a7068;padding:4px 0;">
                <strong style="color:#1a2420;">分析類型</strong>
              </td>
              <td style="font-size:13px;color:#03695e;font-weight:600;text-align:right;padding:4px 0;">
                {type_label}
              </td>
            </tr>
            <tr>
              <td style="font-size:13px;color:#5a7068;padding:4px 0;">
                <strong style="color:#1a2420;">完成時間</strong>
              </td>
              <td style="font-size:13px;color:#5a7068;text-align:right;padding:4px 0;">
                {now_str}
              </td>
            </tr>
          </table>

          <p style="margin:0;font-size:12px;color:#aab8b4;line-height:1.6;">
            此信件由系統自動發送，請勿直接回覆。<br>
            如有問題請聯絡 NCU DentalVis 團隊。
          </p>
        </td></tr>

        <!-- Footer -->
        <tr><td style="background:#f4f6ef;padding:16px 32px;border-top:1px solid #eaede3;">
          <p style="margin:0;font-size:11px;color:#aab8b4;text-align:center;">
            © 2026 DentalVis · National Central University
          </p>
        </td></tr>

      </table>
    </td></tr>
  </table>
</body>
</html>
"""

    text = f"Hi {user_name}，您的{type_label}已完成（{now_str}），請前往 {result_url} 查看結果。"

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"]    = f"{SENDER_NAME} <{SMTP_USER}>"
    msg["To"]      = to_email
    msg.attach(MIMEText(text, "plain", "utf-8"))
    msg.attach(MIMEText(html,  "html",  "utf-8"))

    try:
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as smtp:
            smtp.ehlo()
            smtp.starttls()
            smtp.login(SMTP_USER, SMTP_PASS)
            smtp.sendmail(SMTP_USER, to_email, msg.as_string())
        print(f"  ✅ Email 寄送成功 → {to_email}")
        return True
    except Exception as e:
        print(f"  ⚠️  Email 寄送失敗: {e}")
        return False


def send_analysis_failed(to_email: str, user_name: str, analysis_type: str, error_msg: str = ""):
    """分析失敗通知信"""
    type_label = "牙齒初始化" if analysis_type == "init" else "菌斑分析"
    now_str    = datetime.now().strftime("%Y/%m/%d %H:%M")
    subject    = f"[DentalVis] {type_label}發生錯誤"

    html = f"""
<!DOCTYPE html>
<html lang="zh-TW">
<head><meta charset="UTF-8"></head>
<body style="margin:0;padding:0;background:#f4f6ef;font-family:'DM Sans',Arial,sans-serif;">
  <table width="100%" cellpadding="0" cellspacing="0" style="background:#f4f6ef;padding:40px 0;">
    <tr><td align="center">
      <table width="560" cellpadding="0" cellspacing="0"
             style="background:#ffffff;border-radius:16px;overflow:hidden;box-shadow:0 4px 24px rgba(3,105,94,0.10);">
        <tr><td style="background:#c0392b;padding:28px 32px;">
          <div style="font-size:22px;font-weight:700;color:#ffffff;">DentalVis</div>
          <div style="font-size:13px;color:rgba(255,255,255,0.75);margin-top:4px;">分析失敗通知</div>
        </td></tr>
        <tr><td style="padding:32px;">
          <p style="margin:0 0 16px;font-size:16px;color:#1a2420;">Hi {user_name}，</p>
          <p style="margin:0 0 24px;font-size:15px;color:#5a7068;line-height:1.7;">
            您的 <strong style="color:#c0392b;">{type_label}</strong> 於 {now_str} 發生錯誤，請重新上傳照片再試一次。
          </p>
          <p style="margin:0;font-size:12px;color:#aab8b4;">此信件由系統自動發送，請勿直接回覆。</p>
        </td></tr>
        <tr><td style="background:#f4f6ef;padding:16px 32px;border-top:1px solid #eaede3;">
          <p style="margin:0;font-size:11px;color:#aab8b4;text-align:center;">
            © 2026 DentalVis · National Central University
          </p>
        </td></tr>
      </table>
    </td></tr>
  </table>
</body>
</html>
"""

    text = f"Hi {user_name}，您的{type_label}發生錯誤（{now_str}），請重新嘗試。"

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"]    = f"{SENDER_NAME} <{SMTP_USER}>"
    msg["To"]      = to_email
    msg.attach(MIMEText(text, "plain", "utf-8"))
    msg.attach(MIMEText(html,  "html",  "utf-8"))

    try:
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as smtp:
            smtp.ehlo()
            smtp.starttls()
            smtp.login(SMTP_USER, SMTP_PASS)
            smtp.sendmail(SMTP_USER, to_email, msg.as_string())
        return True
    except Exception as e:
        print(f"  ⚠️  Email 失敗通知寄送失敗: {e}")
        return False


if __name__ == "__main__":
    # 測試
    ok = send_analysis_done("zhen712@g.ncu.edu.tw", "Zhen", "plaque")
    print("test:", "OK" if ok else "FAIL")
