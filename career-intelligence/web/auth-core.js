/* Pure WebCrypto authentication helpers shared by API and Routing Middleware.
   No actual passcode or signing key is committed to source control. */
const encoder=new TextEncoder();
const decoder=new TextDecoder();
const SESSION_SECONDS=12*60*60;
const COOKIE_NAME="hm_session";
const isHex=s=>typeof s==="string"&&/^[a-f0-9]+$/i.test(s)&&s.length%2===0;
function hexBytes(s){if(!isHex(s))return null;const bytes=new Uint8Array(s.length/2);for(let i=0;i<bytes.length;i++)bytes[i]=parseInt(s.slice(2*i,2*i+2),16);return bytes}
function toB64(bytes){return btoa(Array.from(bytes,x=>String.fromCharCode(x)).join("")).replace(/\+/g,"-").replace(/\//g,"_").replace(/=+$/,"")}
function fromB64(str){if(typeof str!=="string"||!/^[a-zA-Z0-9_-]+$/.test(str)||str.length>1000)return null;try{const value=atob(str.replace(/-/g,"+").replace(/_/g,"/")+"===".slice((str.length+3)%4));return Uint8Array.from(value,c=>c.charCodeAt(0))}catch{return null}}
function constantTimeEqual(a,b){if(!a||!b||a.length!==b.length)return false;let diff=0;for(let i=0;i<a.length;i++)diff|=a[i]^b[i];return diff===0}
async function signingKey(secret,usage){const raw=hexBytes(secret);if(!raw||raw.length<32)return null;return crypto.subtle.importKey("raw",raw,{name:"HMAC",hash:"SHA-256"},false,[usage])}
export async function checkPasscode(input,stored){if(typeof input!=="string"||input.length>72||typeof stored!=="string")return false;const [salt,hash]=stored.split(":");const saltBytes=hexBytes(salt),expected=hexBytes(hash);if(!saltBytes||saltBytes.length!==16||!expected||expected.length!==32)return false;const key=await crypto.subtle.importKey("raw",encoder.encode(input),"PBKDF2",false,["deriveBits"]);const derived=new Uint8Array(await crypto.subtle.deriveBits({name:"PBKDF2",hash:"SHA-256",salt:saltBytes,iterations:150000},key,256));return constantTimeEqual(derived,expected)}
export async function issueSession(secret,now=Date.now()){const key=await signingKey(secret,"sign");if(!key)throw Error("Auth secret not configured");const payload=toB64(encoder.encode(JSON.stringify({v:1,e:Math.floor(now/1000)+SESSION_SECONDS})));const sig=toB64(new Uint8Array(await crypto.subtle.sign("HMAC",key,encoder.encode(payload))));return payload+"."+sig}
export async function validateSession(token,secret,now=Date.now()){if(typeof token!=="string"||token.length>600||!token.includes("."))return false;const [payload,sig,...extra]=token.split(".");if(extra.length)return false;const key=await signingKey(secret,"verify"),signature=fromB64(sig),data=fromB64(payload);if(!key||!signature||signature.length!==32||!data)return false;const valid=await crypto.subtle.verify("HMAC",key,signature,encoder.encode(payload));if(!valid)return false;let claims;try{claims=JSON.parse(decoder.decode(data))}catch{return false}return claims?.v===1&&Number.isSafeInteger(claims.e)&&claims.e*1000>now&&claims.e*1000<=now+SESSION_SECONDS*1000+1000}
export function sessionFromCookie(header){if(typeof header!=="string")return null;const match=header.match(/(?:^|;\s*)hm_session=([^;]{1,600})/);return match?match[1]:null}
export const COOKIE=COOKIE_NAME;
export const SESSION_TTL=SESSION_SECONDS;
