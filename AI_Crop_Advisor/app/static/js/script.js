/* ================================================================
   script.js — AI Crop Advisor global JS
   ================================================================ */

// ── Mobile nav toggle ─────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
    const toggle = document.getElementById('navToggle');
    const links  = document.getElementById('navLinks');

    if (toggle && links) {
        toggle.addEventListener('click', () => {
            links.classList.toggle('open');
        });

        // Close on outside click
        document.addEventListener('click', (e) => {
            if (!toggle.contains(e.target) && !links.contains(e.target)) {
                links.classList.remove('open');
            }
        });
    }

    // Auto-dismiss flash messages after 5 s
    document.querySelectorAll('.flash').forEach(el => {
        setTimeout(() => el.remove(), 5000);
    });
});

// ── Utility: format crop name ─────────────────────────────────────
function formatCropName(name) {
    if (!name) return '—';
    return name.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
}

// ── Utility: confidence colour ────────────────────────────────────
function confidenceColour(pct) {
    if (pct >= 75) return 'var(--green-400)';
    if (pct >= 50) return 'var(--amber-400)';
    return '#f87171';
}
