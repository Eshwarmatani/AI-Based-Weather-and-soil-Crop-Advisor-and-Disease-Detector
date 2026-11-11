/* ================================================================
   theme.js — Global Dark / Light Theme Manager
   AI Crop Advisor & Disease Detector
   ================================================================ */

(function () {
    const THEME_STORAGE_KEY = 'agro_theme';

    function getPreferredTheme() {
        const saved = localStorage.getItem(THEME_STORAGE_KEY);
        if (saved === 'light' || saved === 'dark') {
            return saved;
        }
        // Default to dark theme to preserve original aesthetic
        return 'dark';
    }

    function applyTheme(theme, dispatch = true) {
        document.documentElement.setAttribute('data-theme', theme);
        localStorage.setItem(THEME_STORAGE_KEY, theme);
        updateToggleButtonUI(theme);

        if (dispatch) {
            window.dispatchEvent(new CustomEvent('themeChanged', { detail: { theme } }));
        }
    }

    function updateToggleButtonUI(theme) {
        const btn = document.getElementById('themeToggleBtn');
        const sun = document.getElementById('themeIconSun');
        const moon = document.getElementById('themeIconMoon');
        const label = document.getElementById('themeToggleLabel');

        const navBtn = document.getElementById('navThemeToggleBtn');
        const navSun = document.querySelector('.nav-theme-sun');
        const navMoon = document.querySelector('.nav-theme-moon');

        if (theme === 'light') {
            if (btn) {
                btn.setAttribute('title', 'Switch to Dark Mode');
                btn.setAttribute('aria-label', 'Switch to Dark Mode');
            }
            if (sun) sun.style.display = 'none';
            if (moon) moon.style.display = 'inline-block';
            if (label) label.textContent = 'Dark';

            if (navBtn) {
                navBtn.setAttribute('title', 'Switch to Dark Mode');
                navBtn.setAttribute('aria-label', 'Switch to Dark Mode');
            }
            if (navSun) navSun.style.display = 'none';
            if (navMoon) navMoon.style.display = 'inline-block';
        } else {
            if (btn) {
                btn.setAttribute('title', 'Switch to Light Mode');
                btn.setAttribute('aria-label', 'Switch to Light Mode');
            }
            if (sun) sun.style.display = 'inline-block';
            if (moon) moon.style.display = 'none';
            if (label) label.textContent = 'Light';

            if (navBtn) {
                navBtn.setAttribute('title', 'Switch to Light Mode');
                navBtn.setAttribute('aria-label', 'Switch to Light Mode');
            }
            if (navSun) navSun.style.display = 'inline-block';
            if (navMoon) navMoon.style.display = 'none';
        }
    }

    function toggleTheme() {
        const current = document.documentElement.getAttribute('data-theme') || 'dark';
        const next = current === 'dark' ? 'light' : 'dark';
        applyTheme(next);
    }

    // Expose helpers globally
    window.agroTheme = {
        get: () => document.documentElement.getAttribute('data-theme') || 'dark',
        set: applyTheme,
        toggle: toggleTheme
    };

    // Initialize on DOM ready
    document.addEventListener('DOMContentLoaded', () => {
        const current = getPreferredTheme();
        applyTheme(current, false);

        const btn = document.getElementById('themeToggleBtn');
        if (btn) {
            btn.addEventListener('click', toggleTheme);
        }

        const navBtn = document.getElementById('navThemeToggleBtn');
        if (navBtn) {
            navBtn.addEventListener('click', toggleTheme);
        }
    });
})();
