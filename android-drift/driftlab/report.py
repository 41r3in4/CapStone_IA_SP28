import html
from pathlib import Path


def render(out, config, rows, summary):
    points=[]
    for i,r in enumerate(rows):
        if r['f1'] is not None:
            points.append(f'{45+i*690/max(1,len(rows)-1):.1f},{225-190*r["f1"]:.1f}')
    cells=''.join('<tr>'+''.join(f'<td>{html.escape(str(r[k]) if r[k] is not None else "N/A")}</td>' for k in ('month','f1','fnr','fpr','queried_labels','regression','update_seconds'))+'</tr>' for r in rows)
    chart=f'<svg viewBox="0 0 780 255" role="img" aria-label="F1 by month"><path d="M45 25V225H745" stroke="#94a3b8" fill="none"/><text x="8" y="35">1.0</text><text x="8" y="225">0.0</text><polyline points="{" ".join(points)}" fill="none" stroke="#10b981" stroke-width="3"/></svg>'
    text=f'''<!doctype html><html lang="vi"><meta charset="utf-8"><title>Android Drift experiment</title>
<style>body{{font:16px system-ui;background:#0f172a;color:#e2e8f0;max-width:1100px;margin:40px auto;padding:20px}}section{{background:#1e293b;padding:24px;border-radius:12px;margin:20px 0}}table{{width:100%;border-collapse:collapse;font-size:13px}}td,th{{padding:10px;text-align:left;border-bottom:1px solid #475569}}svg{{width:100%;background:#fff;border-radius:8px}}code,pre{{white-space:pre-wrap}}.notice{{color:#fbbf24}}</style>
<h1>Android Drift / Research starter</h1><p class="notice">Nếu dùng data/demo.csv: đây là dữ liệu mô phỏng, không phải kết quả phát hiện APK Android.</p>
<section><h2>{html.escape(config['selector'])} + {html.escape(config['updater'])}</h2><p>Budget: {config['budget']:.0%} | AUT(F1): {summary['aut_f1']} | Queried labels: {summary['queried_labels']}</p>{chart}<p>Trục ngang: các tháng kiểm thử theo thứ tự trong bảng.</p></section>
<section><h2>Test before update</h2><table><tr><th>Month</th><th>F1</th><th>FNR</th><th>FPR</th><th>Labels</th><th>Regression</th><th>Update seconds</th></tr>{cells}</table></section>
<section><h2>Giới hạn và cấu hình</h2><p>Hồi quy được đo sau cập nhật trên monitor quá khứ giữ riêng. Không truy vấn ở tháng cuối. Thời gian phụ thuộc máy chạy.</p><pre>{html.escape(str(summary['notes']))}</pre><pre>{html.escape(str(config))}</pre></section></html>'''
    (Path(out)/'report.html').write_text(text,encoding='utf-8')
