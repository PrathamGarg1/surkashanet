/**
 * Authentication & Parental Session Manager for SurakshaNet
 * Implements PBKDF2-HMAC-SHA256 key derivation and hybrid storage
 * (chrome.storage.sync for resilient credentials, chrome.storage.local for fallback)
 */

const AUTH_CONFIG = {
    storageKey: 'surakshanet_parent_auth',
    sessionKey: 'surakshanet_parent_session',
    pbkdf2Iterations: 100000,
    saltByteLength: 16
};

// In-memory fallback for session state
let inMemorySession = false;

// ─── CRYPTOGRAPHIC UTILITIES (Web Crypto API) ───────────────────────────────

/**
 * Converts ArrayBuffer / Uint8Array to hex string
 */
function bufferToHex(buffer) {
    const bytes = new Uint8Array(buffer);
    return Array.from(bytes).map(b => b.toString(16).padStart(2, '0')).join('');
}

/**
 * Converts hex string to Uint8Array
 */
function hexToBytes(hex) {
    const bytes = new Uint8Array(hex.length / 2);
    for (let i = 0; i < hex.length; i += 2) {
        bytes[i / 2] = parseInt(hex.substring(i, i + 2), 16);
    }
    return bytes;
}

/**
 * Generates a cryptographically random salt hex string
 */
function generateSalt(byteLength = AUTH_CONFIG.saltByteLength) {
    const saltBytes = new Uint8Array(byteLength);
    crypto.getRandomValues(saltBytes);
    return bufferToHex(saltBytes);
}

/**
 * Derives a PBKDF2-HMAC-SHA256 hash from a password and salt
 * @param {string} password - Master password
 * @param {string} saltHex - Salt as hex string
 * @returns {Promise<string>} Hex-encoded derived key (256-bit)
 */
async function hashPassword(password, saltHex) {
    const encoder = new TextEncoder();
    const passwordBuffer = encoder.encode(password);
    const saltBytes = hexToBytes(saltHex);

    const baseKey = await crypto.subtle.importKey(
        'raw',
        passwordBuffer,
        { name: 'PBKDF2' },
        false,
        ['deriveBits']
    );

    const derivedBits = await crypto.subtle.deriveBits(
        {
            name: 'PBKDF2',
            salt: saltBytes,
            iterations: AUTH_CONFIG.pbkdf2Iterations,
            hash: 'SHA-256'
        },
        baseKey,
        256
    );

    return bufferToHex(derivedBits);
}

// ─── STORAGE ACCESS WRAPPERS (Sync with Local Fallback) ─────────────────────

/**
 * Retrieves raw stored auth profile from sync storage (or local fallback)
 */
async function getStoredAuthProfile() {
    try {
        if (chrome?.storage?.sync) {
            const syncData = await chrome.storage.sync.get(AUTH_CONFIG.storageKey);
            if (syncData && syncData[AUTH_CONFIG.storageKey]) {
                return syncData[AUTH_CONFIG.storageKey];
            }
        }
    } catch (e) {
        console.warn('⚠️ SurakshaNet Auth: Failed to read from chrome.storage.sync, falling back to local:', e);
    }

    try {
        if (chrome?.storage?.local) {
            const localData = await chrome.storage.local.get(AUTH_CONFIG.storageKey);
            return localData ? localData[AUTH_CONFIG.storageKey] : null;
        }
    } catch (e) {
        console.error('❌ SurakshaNet Auth: Failed to read local storage:', e);
    }

    return null;
}

/**
 * Saves auth profile to both sync and local storage for redundancy
 */
async function saveAuthProfile(profile) {
    let saved = false;

    try {
        if (chrome?.storage?.sync) {
            await chrome.storage.sync.set({ [AUTH_CONFIG.storageKey]: profile });
            saved = true;
        }
    } catch (e) {
        console.warn('⚠️ SurakshaNet Auth: sync save failed, using local:', e);
    }

    try {
        if (chrome?.storage?.local) {
            await chrome.storage.local.set({ [AUTH_CONFIG.storageKey]: profile });
            saved = true;
        }
    } catch (e) {
        console.error('❌ SurakshaNet Auth: local save failed:', e);
    }

    if (!saved) {
        throw new Error('Could not persist authentication profile to Chrome storage');
    }
}

/**
 * Removes auth profile from both sync and local storage
 */
async function removeAuthProfile() {
    try {
        if (chrome?.storage?.sync) {
            await chrome.storage.sync.remove(AUTH_CONFIG.storageKey);
        }
    } catch (e) { /* ignore */ }

    try {
        if (chrome?.storage?.local) {
            await chrome.storage.local.remove(AUTH_CONFIG.storageKey);
        }
    } catch (e) { /* ignore */ }
}

// ─── PUBLIC AUTHENTICATION APIs ─────────────────────────────────────────────

/**
 * Checks if a Master Parent Account has been created
 * @returns {Promise<boolean>}
 */
export async function isAccountSetup() {
    const profile = await getStoredAuthProfile();
    return !!(profile && profile.passwordHash && profile.salt);
}

/**
 * Sets up a new Master Parent Account
 * @param {Object} params - { email, password }
 * @returns {Promise<boolean>}
 */
export async function setupParentAccount({ email, password }) {
    if (!password || password.length < 6) {
        throw new Error('Master password must be at least 6 characters long');
    }

    const salt = generateSalt();
    const passwordHash = await hashPassword(password, salt);

    const profile = {
        email: email ? email.trim() : '',
        salt: salt,
        passwordHash: passwordHash,
        createdAt: new Date().toISOString(),
        version: '1.0'
    };

    await saveAuthProfile(profile);
    // Automatically activate parent session upon initial setup
    await activateSession();

    console.log('🛡️ SurakshaNet: Master Parent Account successfully configured.');
    return true;
}

/**
 * Verifies a candidate password against the stored master password hash
 * @param {string} password - Candidate password to verify
 * @returns {Promise<boolean>}
 */
export async function verifyParentPassword(password) {
    const profile = await getStoredAuthProfile();
    if (!profile || !profile.passwordHash || !profile.salt) {
        return false;
    }

    const candidateHash = await hashPassword(password, profile.salt);
    return candidateHash === profile.passwordHash;
}

/**
 * Logs in the parent by verifying password and activating the session
 * @param {string} password
 * @returns {Promise<boolean>}
 */
export async function loginParent(password) {
    const isValid = await verifyParentPassword(password);
    if (!isValid) {
        return false;
    }

    await activateSession();
    return true;
}

/**
 * Checks if the current browser session is unlocked in Parent Mode
 * @returns {Promise<boolean>}
 */
export async function isParentSessionActive() {
    try {
        if (chrome?.storage?.session) {
            const data = await chrome.storage.session.get(AUTH_CONFIG.sessionKey);
            return !!(data && data[AUTH_CONFIG.sessionKey]);
        }
    } catch (e) { /* fallback */ }

    return inMemorySession;
}

/**
 * Activates the parent session
 */
async function activateSession() {
    inMemorySession = true;
    try {
        if (chrome?.storage?.session) {
            await chrome.storage.session.set({ [AUTH_CONFIG.sessionKey]: true });
        }
    } catch (e) { /* fallback to memory */ }
}

/**
 * Locks the session back into Child Protection Mode
 */
export async function lockSession() {
    inMemorySession = false;
    try {
        if (chrome?.storage?.session) {
            await chrome.storage.session.remove(AUTH_CONFIG.sessionKey);
        }
    } catch (e) { /* ignore */ }
    console.log('🔒 SurakshaNet: Session locked. Child Protection active.');
}

/**
 * Logs out and resets the parent account (requires master password verification)
 * @param {string} confirmPassword - Parent's master password
 * @returns {Promise<boolean>}
 */
export async function logoutParent(confirmPassword) {
    const isValid = await verifyParentPassword(confirmPassword);
    if (!isValid) {
        throw new Error('Incorrect master password. Logout/Reset denied.');
    }

    await lockSession();
    await removeAuthProfile();
    console.log('🚪 SurakshaNet: Parent account reset/logged out.');
    return true;
}

/**
 * Retrieves parent profile metadata (excluding sensitive hash/salt)
 * @returns {Promise<Object|null>}
 */
export async function getParentProfile() {
    const profile = await getStoredAuthProfile();
    if (!profile) return null;
    return {
        email: profile.email,
        createdAt: profile.createdAt,
        isConfigured: true
    };
}
