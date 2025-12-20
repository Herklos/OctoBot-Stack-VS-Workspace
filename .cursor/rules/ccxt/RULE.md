---
description: "CCXT exchange file coding rules: TypeScript syntax restrictions for transpilation, error handling patterns, and API endpoint management"
globs: ["ccxt/ts/src/**/*.ts", "ccxt/ts/src/abstract/**/*.ts"]
alwaysApply: false
---

# CCXT Exchange File Rules

## Context

Exchange TypeScript files in `ccxt/ts/src/*.ts` are automatically transpiled to Python, Go, PHP, and C#. **NEVER edit transpiled files directly** - only edit TypeScript source files in `ts/src/`.

## Syntax Restrictions (Transpilation Compatibility)

When writing code in exchange files:

**⚠️ CRITICAL: NEVER use ternary operations - they do NOT transpile correctly to Python/Go/PHP/C#**

**NEVER use:**
- **TERNARY OPERATIONS** (`condition ? valueA : valueB`) - **ALWAYS use explicit if/else statements with let variables**
  - ❌ `const value = condition ? valueA : valueB;`
  - ❌ `const value = this.safeInteger (market !== undefined ? this.safeDict (market, 'info', {}) : {}, 'key', 0);` (nested ternary in function call)
  - ❌ `const value = condition1 ? valueA : condition2 ? valueB : valueC;` (nested ternary)
  - ✅ `let value = valueB; if (condition) { value = valueA; }`
  - ✅ `let marketInfo = {}; if (market !== undefined) { marketInfo = this.safeDict (market, 'info', {}); } const value = this.safeInteger (marketInfo, 'key', 0);`
  - **This applies EVERYWHERE: function arguments, variable assignments, return statements, object properties, array elements, etc.**
  - **This is a hard requirement - ternary operators break transpilation**
- **TYPE ANNOTATIONS** - **NEVER use type annotations on variables - they break transpilation**
  - ❌ `let calculatedPrice: number = 0;` (type annotation breaks transpilation)
  - ❌ `let calculatedPrice: number;` (type annotation breaks transpilation)
  - ✅ `let calculatedPrice = 0;` (let TypeScript infer the type)
  - **This is a hard requirement - type annotations break transpilation**
- **EMPTY VARIABLE DECLARATIONS** - **ALWAYS initialize variables with a default value**
  - ❌ `let calculatedPrice;` (empty declaration causes transpilation issues)
  - ✅ `let calculatedPrice = 0;` (initialize with default value)
  - **This is a hard requirement - empty declarations break transpilation**
- **NULL CHECKS** - **NEVER use `=== null` - only check for `undefined`**
  - ❌ `if (value === undefined || value === null)` (null check breaks transpilation)
  - ❌ `if (value === null)` (null check breaks transpilation)
  - ✅ `if (value === undefined)` (only check for undefined)
  - **This is a hard requirement - null checks break transpilation**
- Complex null checks or optional chaining - Use simple if statements
- `any[]` type annotation - Omit type or use proper types like `Dict[]`, `string[]`
- Array mutation methods (`.push()`, `.pop()`, `.shift()`, `.unshift()`) on dictionary values - Build complete arrays conditionally

**ALWAYS:**
- Include a space before opening parenthesis in function calls: `this.safeString (trade, 'id')` not `this.safeString(trade, 'id')`
- Avoid blank lines between consecutive variable declarations (exception: logical groups)

**Examples:**

```typescript
// ❌ BAD: Ternary operations - FORBIDDEN, breaks transpilation
const timestamp = (ts !== undefined) ? ts * 1000 : undefined;
const fee = feeNum !== undefined ? feeNum : this.parseNumber ('0.02');
const symbol = market ? market['symbol'] : undefined;
const sideEnum = (polymarketSide === 'BUY') ? '0' : '1';
const expirationValue = (orderType.toUpperCase () === 'GTD') ? signedOrder['expiration'] : '0';
// ❌ BAD: Ternary in function call arguments
const quoteDecimals = this.safeInteger (market !== undefined ? this.safeDict (market, 'info', {}) : {}, 'quoteDecimals', 6);
// ❌ BAD: Nested ternary
const value = condition1 ? valueA : condition2 ? valueB : valueC;

// ✅ GOOD: Explicit if/else with let variables, space before parenthesis
let timestamp = undefined;
if (ts !== undefined) {
    timestamp = ts * 1000;
}
let fee = this.parseNumber ('0.02');
if (feeNum !== undefined) {
    fee = feeNum;
}
let symbol = undefined;
if (market !== undefined) {
    symbol = market['symbol'];
}
let sideEnum = '1';
if (polymarketSide === 'BUY') {
    sideEnum = '0';
}
let expirationValue = '0';
if (orderType.toUpperCase () === 'GTD') {
    expirationValue = signedOrder['expiration'];
}
// ✅ GOOD: Extract values before function calls
let marketInfo = {};
if (market !== undefined) {
    marketInfo = this.safeDict (market, 'info', {});
}
const quoteDecimals = this.safeInteger (marketInfo, 'quoteDecimals', 6);
// ✅ GOOD: Nested conditions with explicit if/else
let value = valueC;
if (condition1) {
    value = valueA;
} else if (condition2) {
    value = valueB;
}
const id = this.safeString (trade, 'id');

// ❌ BAD: Array mutation methods (don't transpile well to Python)
const types = { 'Order': [] };
types['Order'].push ({ 'name': 'maker', 'type': 'address' });

// ✅ GOOD: Build complete arrays conditionally
const makerField = { 'name': 'maker', 'type': 'address' };
let orderFields = undefined;
if (price !== undefined) {
    orderFields = [ makerField, { 'name': 'price', 'type': 'string' } ];
} else {
    orderFields = [ makerField ];
}
const types = { 'Order': orderFields };
```

## Transpilation Restrictions

When editing exchange files:

**NEVER edit transpiled files directly:**
- `python/ccxt/` - Python transpiled files
- `go/v4/` - Go transpiled files
- `php/ccxt/` - PHP transpiled files
- `cs/ccxt/` - C# transpiled files

**ONLY edit TypeScript source files:**
- `ts/src/*.ts` - TypeScript source files

**NEVER use:**
- Imports from `./static_dependencies/ethers/` - Use base `Exchange` class methods (`hash`, `keccak`, `ecdsa`, `ethEncodeStructuredData`)
  - ❌ `import { TypedDataEncoder } from './static_dependencies/ethers/hash/index.js';`
  - ✅ Use `this.ethEncodeStructuredData()` from base Exchange class (like hyperliquid, dydx do)
- Typed arrays (`Uint8Array`, `Int8Array`, etc.) - Use regular arrays (`[]`)
- Regex literals (`/pattern/`) - Use manual character validation loops
- `charCodeAt()` - Use character comparisons instead

**Examples:**

```typescript
// ❌ BAD: Typed array, regex, ethers import
import { getAddress } from './static_dependencies/ethers/address/address.js';
const expanded = new Uint8Array (40);
const hexPattern = /^[0-9a-f]{40}$/;

// ✅ GOOD: Regular array, manual validation, base methods
const expanded = [];
for (let i = 0; i < address.length; i++) {
    const char = address[i];
    const isDigit = (char >= '0' && char <= '9');
    const isHexLower = (char >= 'a' && char <= 'f');
    if (!(isDigit || isHexLower)) {
        throw new ExchangeError ('Invalid address');
    }
}
this.ethEncodeStructuredData (domain, types, value);
```

## Before Implementing New Functionality

When adding new features to exchange files:

1. **ALWAYS search for similar patterns** in other exchange files (`ts/src/*.ts`) first
2. Use `grep` or `codebase_search` to find examples:
   - Authentication patterns
   - Signing methods
   - Base64 encoding
   - HMAC generation
   - Error handling
3. Follow established patterns for consistency and transpilation compatibility

## Testing Transpilation

When testing exchange changes:

- `npm run transpileRest <exchange_name>` - Python sync/async transpilation
- `npm run transpileWs <exchange_name>` - Python websocket transpilation
- `npm run emitAPI <exchange_name>` - Python abstract transpilation

## API Endpoints - Abstract TypeScript Interface

When adding new API endpoint methods:

**ALWAYS update the abstract file:**
- Update `ts/src/abstract/{exchange}.ts` when adding new API endpoint methods
- Missing endpoint declarations cause TypeScript compilation errors
- Add method signature in same order/location as implementation (public/private, GET/POST/PUT/DELETE)

**Process:**

1. Add to exchange implementation (`ts/src/{exchange}.ts`):
```typescript
async clobPublicPostBooks (params = {}) {
    // implementation
}
```

2. Add to abstract file (`ts/src/abstract/{exchange}.ts`):
```typescript
clobPublicPostBooks (params?: {}): Promise<implicitReturnType>;
```

**Note:** Abstract file location mirrors implementation: `ts/src/abstract/{exchange}.ts` ↔ `ts/src/{exchange}.ts`

## Error Handling

When handling errors in exchange methods:

**NEVER:**
- Catch exceptions in individual methods (`fetchOrder`, `createOrder`, etc.) and access error properties
- Access `error.message` or similar properties (structure differs after transpilation)

**ALWAYS:**
- Handle errors in `handleErrors` method by checking the response body
- Extract error messages from response body: `this.safeString (response, 'error')` or `this.safeString (response, 'message')`
- Use `handleErrors` which receives parsed `response` and raw `body` (consistent across languages)

**Error handling pattern:**

```typescript
// ❌ BAD: Catching in fetchOrder
async fetchOrder (id: string, symbol: Str = undefined, params = {}) {
    try {
        const response = await this.clobPrivateGetOrder ({ 'order_id': id }, params);
        return this.parseOrder (response, market);
    } catch (error) {
        if (error instanceof BadRequest) {
            const errorMessage = error.message; // ❌ Fails in Python!
            if (errorMessage.indexOf ('Invalid orderID') >= 0) {
                throw new OrderNotFound (this.id + ' order ' + id + ' not found');
            }
        }
        throw error;
    }
}

// ✅ GOOD: Handle in handleErrors
// In describe() - add to exceptions mapping:
'exceptions': {
    'exact': { 'Invalid orderID': OrderNotFound },
    'broad': { 'not found': ExchangeError },
},

// In handleErrors():
handleErrors (code: int, reason: string, url: string, method: string, headers: Dict, body: string, response: any, requestHeaders: any, requestBody: any) {
    if (response === undefined) return undefined;
    if (code >= 400) {
        const errorMessage = this.safeString (response, 'error'); // Extract from response body
        if (errorMessage !== undefined) {
            const feedback = this.id + ' ' + body;
            this.throwExactlyMatchedException (this.exceptions['exact'], errorMessage, feedback);
            this.throwBroadlyMatchedException (this.exceptions['broad'], errorMessage, feedback);
            throw new ExchangeError (feedback);
        }
    }
    return undefined;
}

// In fetchOrder() - no try-catch needed:
async fetchOrder (id: string, symbol: Str = undefined, params = {}) {
    await this.loadMarkets ();
    const response = await this.clobPrivateGetOrder (this.extend ({ 'order_id': id }, params));
    return this.parseOrder (response, market);
}
```

**Error handling guidelines:**
- Add exact matches to `exceptions['exact']`: `'Invalid orderID': OrderNotFound`
- Add patterns to `exceptions['broad']`: `'not found': ExchangeError`
- Matching order: exact first, then broad
- Reference examples: `ts/src/dydx.ts`, `ts/src/hyperliquid.ts`, `ts/src/cex.ts`

## Precision Handling

When converting decimal amounts to smallest units (e.g., wei, satoshi):

**NEVER:**
- Hardcode precision values (`'1000000'`, `'1000000000000000000'`)
- Use `Math.random()` (use `this.microseconds()` instead)
- Manually construct multipliers (`'1' + '0'.repeat(decimals)`)

**ALWAYS:**
- Parse precision from market: `market.info.quoteDecimals`/`baseDecimals` or `market.precision.price`/`amount`
- Use `integerPrecisionToAmount(-decimals)` to get multiplier (10^decimals)
- Convert to integer: `Precise.stringDiv(Precise.stringMul(amount, multiplier), '1', 0)`

```typescript
// ❌ BAD
const USDC_DECIMALS = '1000000';
const multiplier = '1' + '0'.repeat(decimals);

// ✅ GOOD
const quoteDecimals = this.safeInteger(market.info, 'quoteDecimals', this.safeInteger(market.precision, 'price', 6));
const multiplier = this.integerPrecisionToAmount(this.numberToString(-quoteDecimals));
const amountInSmallestUnit = Precise.stringDiv(Precise.stringMul(amount, multiplier), '1', 0);
```

**Store precision in market:**
```typescript
'precision': { 'amount': 18, 'price': 6 },
'info': { 'quoteDecimals': 6, 'baseDecimals': 18 },
```
