import { next } from "@vercel/functions";
import {sessionFromCookie,validateSession} from "./auth-core.js";
const PUBLIC=new Set(["/login.html","/login.css","/login.js","/api/auth","/favicon.ico"]);
const locked=()=>({"Cache-Control":"no-store","X-Robots-Tag":"noindex, nofollow","Referrer-Policy":"no-referrer"});
export const config={matcher:"/:path*"};
export default async function middleware(request){
 const url=new URL(request.url),path=url.pathname;
 if(PUBLIC.has(path))return next();
 const token=sessionFromCookie(request.headers.get("cookie"));
 if(await validateSession(token,process.env.HM_AUTH_SESSION_SECRET))return next();
 if(path.startsWith("/api/"))return new Response(JSON.stringify({ok:false,error:"Authentication required"}),{status:401,headers:{"Content-Type":"application/json; charset=utf-8",...locked()}});
 if(request.method!=="GET"&&request.method!=="HEAD")return new Response("Authentication required",{status:401,headers:locked()});
 const login=new URL("/login.html",url);
 if(path!=="/")login.searchParams.set("next",path+(url.search||""));
 return new Response(null,{status:302,headers:{"Location":login.pathname+login.search,...locked()}});
}
