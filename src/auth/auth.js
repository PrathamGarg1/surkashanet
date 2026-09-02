import { isAccountSetup, setupParentAccount, loginParent, getParentProfile } from '../utils/auth_manager.js';

// DOM Elements
const tabSignup = document.getElementById('tabSignup');
const tabLogin = document.getElementById('tabLogin');
const signupForm = document.getElementById('signupForm');
const loginForm = document.getElementById('loginForm');
const alertBox = document.getElementById('alertBox');

const signupEmail = document.getElementById('signupEmail');
const signupPassword = document.getElementById('signupPassword');
const signupConfirmPassword = document.getElementById('signupConfirmPassword');
const strengthBar = document.getElementById('strengthBar');
const strengthText = document.getElementById('strengthText');
const btnSignupSubmit = document.getElementById('btnSignupSubmit');

const loginPassword = document.getElementById('loginPassword');
const btnLoginSubmit = document.getElementById('btnLoginSubmit');

// ─── INITIALIZATION ─────────────────────────────────────────────────────────

document.addEventListener('DOMContentLoaded', async () => {
    setupEventListeners();
    await checkInitialState();
});

async function checkInitialState() {
    const isConfigured = await isAccountSetup();
    if (isConfigured) {
        // Account exists: switch to Login tab
        switchTab('login');
        const profile = await getParentProfile();
        if (profile?.email) {
            showAlert(`Registered parent: <strong>${profile.email}</strong>. Please log in below.`, 'success');
        }
    } else {
        // First-time setup: switch to Signup tab
        switchTab('signup');
    }
}

function setupEventListeners() {
    tabSignup.addEventListener('click', () => switchTab('signup'));
    tabLogin.addEventListener('click', () => switchTab('login'));

    // Password visibility toggles
    document.querySelectorAll('.toggle-pwd').forEach(btn => {
        btn.addEventListener('click', () => {
            const targetId = btn.getAttribute('data-target');
            const input = document.getElementById(targetId);
            if (input) {
                if (input.type === 'password') {
                    input.type = 'text';
                    btn.textContent = '🙈';
                } else {
                    input.type = 'password';
                    btn.textContent = '👁️';
                }
            }
        });
    });

    // Password strength evaluator
    signupPassword.addEventListener('input', () => {
        evaluatePasswordStrength(signupPassword.value);
    });

    // Form submissions
    signupForm.addEventListener('submit', handleSignup);
    loginForm.addEventListener('submit', handleLogin);
}

// ─── TAB SWITCHING ──────────────────────────────────────────────────────────

function switchTab(mode) {
    hideAlert();
    if (mode === 'signup') {
        tabSignup.classList.add('active');
        tabLogin.classList.remove('active');
        signupForm.style.display = 'block';
        loginForm.style.display = 'none';
    } else {
        tabLogin.classList.add('active');
        tabSignup.classList.remove('active');
        loginForm.style.display = 'block';
        signupForm.style.display = 'none';
    }
}

// ─── PASSWORD STRENGTH ──────────────────────────────────────────────────────

function evaluatePasswordStrength(password) {
    let score = 0;
    if (!password) {
        strengthBar.style.width = '0%';
        strengthBar.style.backgroundColor = 'transparent';
        strengthText.textContent = 'Password strength: Empty';
        return;
    }

    if (password.length >= 6) score += 25;
    if (password.length >= 10) score += 25;
    if (/[A-Z]/.test(password)) score += 15;
    if (/[0-9]/.test(password)) score += 15;
    if (/[^A-Za-z0-9]/.test(password)) score += 20;

    let color = '#ef4444'; // Red
    let label = 'Weak';

    if (score >= 75) {
        color = '#10b981'; // Green
        label = 'Strong';
    } else if (score >= 50) {
        color = '#f59e0b'; // Orange
        label = 'Medium';
    }

    strengthBar.style.width = `${score}%`;
    strengthBar.style.backgroundColor = color;
    strengthText.innerHTML = `Password strength: <strong style="color: ${color}">${label}</strong>`;
}

// ─── ALERTS & FEEDBACK ──────────────────────────────────────────────────────

function showAlert(message, type = 'error') {
    alertBox.innerHTML = message;
    alertBox.className = `alert-box alert-${type}`;
    alertBox.style.display = 'block';
}

function hideAlert() {
    alertBox.style.display = 'none';
}

// ─── FORM HANDLERS ──────────────────────────────────────────────────────────

async function handleSignup(e) {
    e.preventDefault();
    hideAlert();

    const email = signupEmail.value.trim();
    const password = signupPassword.value;
    const confirmPassword = signupConfirmPassword.value;

    if (password.length < 6) {
        showAlert('Password must be at least 6 characters long.', 'error');
        return;
    }

    if (password !== confirmPassword) {
        showAlert('Passwords do not match. Please verify and try again.', 'error');
        return;
    }

    try {
        btnSignupSubmit.disabled = true;
        btnSignupSubmit.textContent = '⏳ Setting Up Protection...';

        await setupParentAccount({ email, password });

        showAlert('🎉 <strong>Success!</strong> Master Parent Account created. Parental protection is now fully activated across WhatsApp Web, Twitter/X, and Instagram.', 'success');
        
        btnSignupSubmit.textContent = '✅ Protection Activated!';
        
        setTimeout(() => {
            // If opened in a tab, let parent know they can close this tab or open the extension popup
            btnSignupSubmit.textContent = '🛡️ Setup Complete (You can close this tab)';
        }, 2000);
    } catch (error) {
        console.error('Setup failed:', error);
        showAlert(`Failed to set up account: ${error.message}`, 'error');
        btnSignupSubmit.disabled = false;
        btnSignupSubmit.textContent = '🛡️ Activate Parental Protection';
    }
}

async function handleLogin(e) {
    e.preventDefault();
    hideAlert();

    const password = loginPassword.value;

    try {
        btnLoginSubmit.disabled = true;
        btnLoginSubmit.textContent = '⏳ Verifying...';

        const success = await loginParent(password);

        if (success) {
            showAlert('🔓 <strong>Parent Mode Unlocked!</strong> You can now open the SurakshaNet toolbar extension popup to view evidence logs and reports.', 'success');
            btnLoginSubmit.textContent = '✅ Unlocked!';
            setTimeout(() => {
                loginPassword.value = '';
                btnLoginSubmit.disabled = false;
                btnLoginSubmit.textContent = '🔓 Unlock Parent Dashboard';
            }, 3000);
        } else {
            showAlert('❌ Incorrect master password. Access denied.', 'error');
            btnLoginSubmit.disabled = false;
            btnLoginSubmit.textContent = '🔓 Unlock Parent Dashboard';
        }
    } catch (error) {
        console.error('Login error:', error);
        showAlert(`Authentication error: ${error.message}`, 'error');
        btnLoginSubmit.disabled = false;
        btnLoginSubmit.textContent = '🔓 Unlock Parent Dashboard';
    }
}
