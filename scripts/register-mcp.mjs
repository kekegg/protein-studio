#!/usr/bin/env node
/**
 * Register an MCP server with one or more MCP clients.
 *
 * Usage A — register the bundled protein-studio server with all clients:
 *
 *   node scripts/register-mcp.mjs
 *   node scripts/register-mcp.mjs all
 *   node scripts/register-mcp.mjs claude-desktop
 *
 * Usage B — register a custom server in compact "<name>:<command> [args...]"
 * shorthand against one or more clients (you can repeat --client many times):
 *
 *   node scripts/register-mcp.mjs \
 *       --client claude-desktop \
 *       protein-studio:/opt/homebrew/.../python -m protein_studio.mcp_server
 *
 * The compact form lets you paste a single-line definition (the same
 * syntax you'd see in Claude Desktop's "command" column), and the
 * script will fill in cwd, PYTHONPATH and QT_QPA_PLATFORM=offscreen
 * automatically (overridable via env vars).
 *
 * Supported clients:
 *
 *   claude-desktop   writes  ~/Library/Application Support/Claude/claude_desktop_config.json
 *   cline            writes  ~/.config/Code/.../cline_mcp_settings.json
 *   cherry-studio    writes  ~/.config/cherry-studio/mcp_settings.json
 *   continue         writes  ~/.continue/config.json
 *   all              registers with all four (default when no client arg)
 *
 * No deps — pure stdlib (fs, os, path).
 */

import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';

const PROJECT_ROOT = path.resolve(
    path.dirname(new URL(import.meta.url).pathname), '..');
const DEFAULT_PYTHON =
    '/opt/homebrew/Caskroom/miniconda/base/envs/protein-studio/bin/python';

const CLIENTS = {
    'claude-desktop': () => path.join(
        os.homedir(),
        'Library/Application Support/Claude/claude_desktop_config.json'),
    cline: () => {
        const override = process.env.CLINE_SETTINGS;
        if (override) return override;
        return path.join(os.homedir(),
            '.config/Code/User/globalStorage/saoudrizwan.claude-dev/'
          + 'settings/cline_mcp_settings.json');
    },
    'cherry-studio': () => {
        const override = process.env.CHERRY_STUDIO_SETTINGS;
        if (override) return override;
        return path.join(os.homedir(),
            '.config/cherry-studio/mcp_settings.json');
    },
    continue: () => path.join(os.homedir(), '.continue/config.json'),
};

// ----- compact-form <name>:<command> [args...] parser ----------------

/**
 * Parse a single compact-form definition. The first ':' separates the
 * server name from the command line; everything after is split on
 * whitespace into [command, arg1, arg2, ...].
 *
 * Examples:
 *   "protein-studio:/opt/.../python -m protein_studio.mcp_server"
 *   "filesystem:uvx mcp-server-filesystem /tmp"
 *   "pymol:/opt/.../python /Users/.../pymol_mcp_server.py"
 *
 * If no ':' is present, the whole string is treated as the command and
 * the server name defaults to "protein-studio".
 */
function parseCompactForm(token) {
    const colon = token.indexOf(':');
    let name, rest;
    if (colon === -1) {
        name = 'protein-studio';
        rest = token;
    } else {
        name = token.slice(0, colon).trim();
        rest = token.slice(colon + 1).trim();
    }
    // Split on whitespace, respecting simple double-quoted segments.
    const args = [];
    let cur = '';
    let inQ = false;
    for (let i = 0; i < rest.length; i++) {
        const c = rest[i];
        if (c === '"') { inQ = !inQ; continue; }
        if (!inQ && /\s/.test(c)) {
            if (cur) { args.push(cur); cur = ''; }
            continue;
        }
        cur += c;
    }
    if (cur) args.push(cur);
    if (!name) throw new Error(`bad compact form (no name): ${token}`);
    if (args.length === 0) {
        throw new Error(`bad compact form (no command): ${token}`);
    }
    return { name, command: args[0], args: args.slice(1) };
}

// ----- protein-studio default entry ---------------------------------

function proteinStudioCompact() {
    const py = process.env.PROTEIN_STUDIO_PYTHON || DEFAULT_PYTHON;
    return `${SERVER_KEY_PROTEIN_STUDIO}:${py} -m protein_studio.mcp_server`;
}

function proteinStudioEntry() {
    const { name, command, args } = parseCompactForm(proteinStudioCompact());
    return {
        name,
        command,
        args,
        cwd: PROJECT_ROOT,
        env: {
            PYTHONPATH: path.join(PROJECT_ROOT, 'src'),
            QT_QPA_PLATFORM: 'offscreen',
        },
    };
}

const SERVER_KEY_PROTEIN_STUDIO = 'protein-studio';

// ----- generic MCP entry from compact form --------------------------

function entryFromCompact(token) {
    const parsed = parseCompactForm(token);
    return {
        name: parsed.name,
        command: parsed.command,
        args: parsed.args,
        cwd: process.env.MCP_CWD || PROJECT_ROOT,
        env: {
            PYTHONPATH: path.join(
                process.env.MCP_CWD || PROJECT_ROOT, 'src'),
            QT_QPA_PLATFORM: 'offscreen',
        },
    };
}

// ----- file I/O ------------------------------------------------------

function readJsonOrEmpty(p) {
    if (!fs.existsSync(p)) return {};
    try {
        const raw = fs.readFileSync(p, 'utf-8').trim();
        if (!raw) return {};
        return JSON.parse(raw);
    } catch (e) {
        throw new Error(`could not parse ${p}: ${e.message}`);
    }
}

function writeJsonAtomic(p, obj) {
    fs.mkdirSync(path.dirname(p), { recursive: true });
    const tmp = `${p}.tmp-${process.pid}`;
    fs.writeFileSync(tmp, JSON.stringify(obj, null, 2) + '\n');
    fs.renameSync(tmp, p);
}

// ----- registration ---------------------------------------------------

function registerOne(clientName, factory, entry) {
    const configPath = factory();
    const cfg = readJsonOrEmpty(configPath);
    if (!cfg.mcpServers || typeof cfg.mcpServers !== 'object') {
        cfg.mcpServers = {};
    }
    const had = entry.name in cfg.mcpServers;
    cfg.mcpServers[entry.name] = {
        command: entry.command,
        args: entry.args,
        cwd: entry.cwd,
        env: entry.env,
    };
    writeJsonAtomic(configPath, cfg);
    return { configPath, had };
}

// ----- main ----------------------------------------------------------

function parseArgs(argv) {
    const positional = [];
    const clients = new Set();
    for (let i = 2; i < argv.length; i++) {
        const a = argv[i];
        if (a === '--client' && argv[i + 1]) {
            clients.add(argv[++i]);
        } else if (a.startsWith('--client=')) {
            clients.add(a.slice('--client='.length));
        } else if (a === '--help' || a === '-h') {
            console.log(HELP);
            process.exit(0);
        } else {
            positional.push(a);
        }
    }
    return { positional, clients };
}

const HELP = `
Usage:
  node scripts/register-mcp.mjs [all|<client>]                 # bundled protein-studio
  node scripts/register-mcp.mjs --client <client> [--client ...] \\
                                            "<name>:<command> [args...]"

Examples:
  node scripts/register-mcp.mjs all
  node scripts/register-mcp.mjs claude-desktop
  node scripts/register-mcp.mjs --client claude-desktop \\
      protein-studio:/opt/.../python -m protein_studio.mcp_server

  # Show the bundled one-liner (no registration):
  node scripts/register-mcp.mjs --print
`;

function main() {
    const { positional, clients } = parseArgs(process.argv);

    // --print: just emit the bundled protein-studio compact form.
    if (positional.includes('--print') || process.argv.includes('--print')) {
        console.log(proteinStudioCompact());
        return;
    }

    let entry;
    if (positional.length === 0) {
        // Default: bundled protein-studio server.
        entry = proteinStudioEntry();
    } else {
        // Compact form: collapse all positionals into one definition.
        // We allow splitting across argv for shell-friendliness, but
        // if the first token contains ':' we treat that as the split
        // point and join the rest verbatim.
        const first = positional[0];
        if (first.includes(':')) {
            entry = entryFromCompact(positional.join(' '));
        } else {
            // No name prefix — assume bundled protein-studio.
            entry = proteinStudioEntry();
        }
    }

    // Decide which clients to register with.
    let names;
    if (clients.size > 0) {
        names = [...clients];
    } else if (positional.length === 0 || positional[0] === 'all') {
        names = Object.keys(CLIENTS);
    } else if (CLIENTS[positional[0]]) {
        names = [positional[0]];
    } else {
        // Compact-form with no explicit client → register with all.
        names = Object.keys(CLIENTS);
    }
    // Expand any "all" pseudo-name into the real client list.
    if (names.includes('all')) {
        names = Object.keys(CLIENTS);
    }

    for (const name of names) {
        const factory = CLIENTS[name];
        if (!factory) {
            console.error(`unknown client: ${name}`);
            console.error(`known: ${Object.keys(CLIENTS).join(', ')}, all`);
            process.exit(2);
        }
        const { configPath, had } = registerOne(name, factory, entry);
        const verb = had ? 'updated' : 'added';
        console.log(`  ${verb}  ${name.padEnd(15)}  ${configPath}`);
    }

    console.log('\nCompact form:');
    console.log(`  ${entry.name}:${entry.command} ${entry.args.join(' ')}`);
    console.log('\nRestart the client to pick up the new server.');
}

main();