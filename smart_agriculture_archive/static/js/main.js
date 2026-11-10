/**
 * Smart Agriculture - Main JavaScript
 */

document.addEventListener('DOMContentLoaded', function () {
    // 1. Sidebar Toggle for Mobile
    const sidebarToggle = document.getElementById('sidebar-toggle');
    const sidebarWrapper = document.getElementById('sidebar-wrapper');
    if (sidebarToggle && sidebarWrapper) {
        sidebarToggle.addEventListener('click', function () {
            sidebarWrapper.classList.toggle('toggled');
        });
    }

    // Close sidebar when clicking outside on mobile
    document.addEventListener('click', function (event) {
        if (window.innerWidth < 992 && sidebarWrapper && sidebarWrapper.classList.contains('toggled')) {
            if (!sidebarWrapper.contains(event.target) && event.target !== sidebarToggle && !sidebarToggle.contains(event.target)) {
                sidebarWrapper.classList.remove('toggled');
            }
        }
    });

    // 2. Initialize Bootstrap Tooltips if available
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });

    // 3. Quick Prefill Sample Data Handlers
    window.fillSample = function (preset) {
        const presets = {
            'rice': {
                'nitrogen': 80, 'phosphorus': 48, 'potassium': 40,
                'temperature': 24, 'humidity': 82, 'ph': 6.5, 'rainfall': 220
            },
            'chickpea': {
                'nitrogen': 40, 'phosphorus': 68, 'potassium': 80,
                'temperature': 19, 'humidity': 17, 'ph': 7.3, 'rainfall': 80
            },
            'coffee': {
                'nitrogen': 100, 'phosphorus': 28, 'potassium': 30,
                'temperature': 25, 'humidity': 60, 'ph': 6.7, 'rainfall': 160
            },
            'apple': {
                'nitrogen': 25, 'phosphorus': 130, 'potassium': 200,
                'temperature': 22, 'humidity': 92, 'ph': 6.0, 'rainfall': 110
            },
            'acidic_soil': {
                'ph': 4.8, 'nitrogen': 25, 'phosphorus': 15,
                'potassium': 20, 'moisture': 35, 'temperature': 26
            },
            'optimal_soil': {
                'ph': 6.8, 'nitrogen': 75, 'phosphorus': 50,
                'potassium': 55, 'moisture': 65, 'temperature': 25
            },
            'alkaline_soil': {
                'ph': 8.8, 'nitrogen': 30, 'phosphorus': 18,
                'potassium': 22, 'moisture': 20, 'temperature': 33
            }
        };

        const data = presets[preset];
        if (!data) return;

        for (const [key, value] of Object.entries(data)) {
            const el = document.getElementById(key) || document.querySelector(`[name="${key}"]`);
            if (el) {
                el.value = value;
                el.classList.add('border-success');
                setTimeout(() => el.classList.remove('border-success'), 1200);
            }
        }
    };
});
