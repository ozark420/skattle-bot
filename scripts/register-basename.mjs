#!/usr/bin/env node
/**
 * Basename Registration Script
 * Registers .base.eth names for AI agent wallets
 * 
 * Usage:
 *   node register-basename.mjs --check skattlebot       # Check availability
 *   node register-basename.mjs skattlebot               # Register for 1 year
 * 
 * Environment:
 *   NET_PRIVATE_KEY - Your wallet private key (0x prefixed)
 */

import { createWalletClient, createPublicClient, http } from 'viem';
import { base } from 'viem/chains';
import { privateKeyToAccount } from 'viem/accounts';

const REGISTRAR = '0xa7d2607c6BD39Ae9521e514026CBB078405Ab322';
const RESOLVER = '0x426fA03fB86E510d0Dd9F70335Cf102a98b10875';

const ABI = [
  {
    name: 'register',
    type: 'function',
    inputs: [{
      name: 'request',
      type: 'tuple',
      components: [
        { name: 'name', type: 'string' },
        { name: 'owner', type: 'address' },
        { name: 'duration', type: 'uint256' },
        { name: 'resolver', type: 'address' },
        { name: 'data', type: 'bytes[]' },
        { name: 'reverseRecord', type: 'bool' },
        { name: 'coinTypes', type: 'uint256[]' },
        { name: 'signatureExpiry', type: 'uint256' },
        { name: 'signature', type: 'bytes' }
      ]
    }],
    outputs: [],
    stateMutability: 'payable'
  },
  {
    name: 'registerPrice',
    type: 'function',
    inputs: [
      { name: 'name', type: 'string' },
      { name: 'duration', type: 'uint256' }
    ],
    outputs: [{ name: '', type: 'uint256' }],
    stateMutability: 'view'
  },
  {
    name: 'available',
    type: 'function',
    inputs: [{ name: 'name', type: 'string' }],
    outputs: [{ name: '', type: 'bool' }],
    stateMutability: 'view'
  }
];

const SECONDS_PER_YEAR = 31536000n;

async function main() {
  const args = process.argv.slice(2);
  const checkOnly = args.includes('--check');
  const name = args.find(a => !a.startsWith('--')) || 'skattlebot';

  const publicClient = createPublicClient({
    chain: base,
    transport: http('https://mainnet.base.org')
  });

  const isAvailable = await publicClient.readContract({
    address: REGISTRAR,
    abi: ABI,
    functionName: 'available',
    args: [name]
  });

  console.log(`Name: ${name}.base.eth`);
  console.log(`Available: ${isAvailable ? 'YES' : 'NO (already taken)'}`);

  if (!isAvailable) process.exit(1);

  const duration = SECONDS_PER_YEAR;
  const price = await publicClient.readContract({
    address: REGISTRAR,
    abi: ABI,
    functionName: 'registerPrice',
    args: [name, duration]
  });

  console.log(`Price: ${(Number(price) / 1e18).toFixed(6)} ETH (~$${(Number(price) / 1e18 * 2500).toFixed(2)})`);

  if (checkOnly) {
    console.log('\nRun without --check to register.');
    process.exit(0);
  }

  const privateKey = process.env.NET_PRIVATE_KEY;
  if (!privateKey) {
    console.error('\nNET_PRIVATE_KEY required. Run:');
    console.error('$env:NET_PRIVATE_KEY="0x..."; node register-basename.mjs skattlebot');
    process.exit(1);
  }

  const account = privateKeyToAccount(privateKey);
  console.log(`\nWallet: ${account.address}`);

  const walletClient = createWalletClient({
    account,
    chain: base,
    transport: http('https://mainnet.base.org')
  });

  const paymentAmount = price * 150n / 100n;
  console.log(`Registering ${name}.base.eth...`);

  const hash = await walletClient.writeContract({
    address: REGISTRAR,
    abi: ABI,
    functionName: 'register',
    args: [{
      name,
      owner: account.address,
      duration,
      resolver: RESOLVER,
      data: [],
      reverseRecord: true,
      coinTypes: [],
      signatureExpiry: 0n,
      signature: '0x'
    }],
    value: paymentAmount,
    gas: 500000n
  });

  console.log(`TX: ${hash}`);
  const receipt = await publicClient.waitForTransactionReceipt({ hash });
  console.log(receipt.status === 'reverted' ? 'FAILED' : `SUCCESS! ${name}.base.eth registered`);
}

main().catch(console.error);
