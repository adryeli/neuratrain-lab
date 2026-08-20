"""Self-contained completion celebration for notebooks.

No external CDN dependency: the confetti effect is a small inline
``<canvas>`` + ``requestAnimationFrame`` animation, so it keeps working
offline. IPython is imported lazily inside ``celebrate`` so the Streamlit
app / Docker image never need it on their import path.
"""

from __future__ import annotations

import uuid

PALETTE = ["#7C3AED", "#2563EB", "#F97316", "#22C55E", "#F43F5E", "#FACC15"]


def celebrate(message: str, subtext: str = "", n_pieces: int = 140):
    """Return an ``IPython.display.HTML`` confetti banner for a notebook's last cell.

    The message/subtext render even if the script does not execute (e.g.
    headless ``nbconvert``), so the congratulations text is never lost —
    only the animation itself needs a live browser context.
    """

    from IPython.display import HTML

    box_id = f"celebrate-{uuid.uuid4().hex[:8]}"
    colors_js = ", ".join(f'"{c}"' for c in PALETTE)
    return HTML(
        f"""
        <div id="{box_id}" style="position:relative; overflow:hidden; text-align:center;
             padding:1.6rem; border-radius:1rem; color:white; font-family:sans-serif;
             background:linear-gradient(120deg,#312E81 0%,#7C3AED 55%,#2563EB 100%);">
          <canvas id="{box_id}-canvas" style="position:absolute; inset:0; pointer-events:none;"></canvas>
          <div style="position:relative; z-index:2;">
            <h2 style="margin:0;">{message}</h2>
            <p style="margin:.4rem 0 0; opacity:.9;">{subtext}</p>
          </div>
        </div>
        <script>
        (function() {{
          var box = document.getElementById("{box_id}");
          var canvas = document.getElementById("{box_id}-canvas");
          if (!box || !canvas || !canvas.getContext) return;
          var w = box.offsetWidth || 600, h = box.offsetHeight || 160;
          canvas.width = w; canvas.height = h;
          var ctx = canvas.getContext("2d");
          var colors = [{colors_js}];
          var pieces = [];
          for (var i = 0; i < {n_pieces}; i++) {{
            pieces.push({{
              x: Math.random() * w, y: -20 - Math.random() * h,
              size: 4 + Math.random() * 5, speed: 1.5 + Math.random() * 3,
              drift: (Math.random() - 0.5) * 2, rot: Math.random() * 360,
              spin: (Math.random() - 0.5) * 10, color: colors[i % colors.length]
            }});
          }}
          var frame = 0, maxFrames = 260;
          function tick() {{
            frame++;
            ctx.clearRect(0, 0, w, h);
            for (var i = 0; i < pieces.length; i++) {{
              var p = pieces[i];
              p.y += p.speed; p.x += p.drift; p.rot += p.spin;
              ctx.save(); ctx.translate(p.x, p.y); ctx.rotate(p.rot * Math.PI / 180);
              ctx.fillStyle = p.color; ctx.fillRect(-p.size / 2, -p.size / 2, p.size, p.size * 0.6);
              ctx.restore();
            }}
            if (frame < maxFrames) {{ window.requestAnimationFrame(tick); }}
            else {{ ctx.clearRect(0, 0, w, h); }}
          }}
          window.requestAnimationFrame(tick);
        }})();
        </script>
        """
    )
