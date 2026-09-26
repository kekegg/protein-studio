#!/usr/bin/env node
/**
 * Register the LingSu Protein MCP server with an MCP client.
 *
 * Usage:
 *   node scripts/register-mcp.mjs <client>
 *
 * Where <client> is one of:
 *
 *   claude-desktop   writes  ~/Library/Application Support/Claude/claude_desktop_config.json
 *   cline            writes  ~/.config/Code/User/globalStorage/saoudrizwan.claude-dev/settings/cline_mcp_settings.json
 *   cherry-studio    writes  ~/.config/cherry-studio/mcp_settings.json
 *   continue         writes  ~/.continue/config.json
 *   all              registers with all four
 *
 * The script:
 *   - locates the config file (creating it if missing)
 *   - injects an entry under mcpServers["protein-studio"]
 *   - uses atomic write (temp + rename) so a half-finished config
 *     never overwrites the existing one
 *   - leaves existing servers untouched — re-running the script just
 *     refreshes our entry
 *
 * No deps — pure stdlib (fs, os, path).
 */

import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';

const SERVER_KEY = 'protein-studio';

function proteinStudioEntry() {
    const projectRoot = path.resolve(
        path.dirname(new URL(import.meta.url).pathname), '..');
    return {
        command: process.env.PROTEIN_STUDIO_PYTHON
                  || '/opt/homebrew/Caskroom/miniconda/base/envs/protein-studio/bin/python',
        args: ['-m', 'protein_studio.mcp_server'],
        cwd: projectRoot,
        env: {
            PYTHONPATH: path.join(projectRoot, 'src'),
            QT_QPA_PLATFORM: 'offscreen',
        },
    };
}

const CLIENTS = {
    'claude-desktop': () => path.join(
        os.homedir(),
        'Library/Application Support/Claude/claude_desktop_config.json'),

    cline: () => {
        // macOS path; on Linux/Windows the user can override with the
        // CLINE_SETTINGS env var if needed.
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

function registerOne(clientName, factory) {
    const configPath = factory();
    const cfg = readJsonOrEmpty(configPath);
    if (!cfg.mcpServers || typeof cfg.mcpServers !== 'object') {
        cfg.mcpServers = {};
    }
    const had = SERVER_KEY in cfg.mcpServers;
    cfg.mcpServers[SERVER_KEY] = proteinStudioEntry();
    writeJsonAtomic(configPath, cfg);
    return { configPath, had };
}

function main() {
    const arg = process.argv[2] || 'all';
    const names = arg === 'all'
        ? Object.keys(CLIENTS)
        : [arg];

    for (const name of names) {
        const factory = CLIENTS[name];
        if (!factory) {
            console.error(`unknown client: ${name}`);
            console.error(`known: ${Object.keys(CLIENTS).join(', ')}, all`);
            process.exit(2);
        }
        const { configPath, had } = registerOne(name, factory);
        const verb = had ? 'updated' : 'added';
        console.log(`  ${verb}  ${name.padEnd(15)}  ${configPath}`);
    }

    const projectRoot = path.resolve(
        path.dirname(new URL(import.meta.url).pathname), '..');
    console.log('\nMCP server entry points to:');
    console.log(`  command: ${proteinStudioEntry().command}`);
    console.log(`  cwd:     ${projectRoot}`);
    console.log('\nRestart the client to pick up the new server.');
}

main();