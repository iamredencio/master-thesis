import {checkPasscode,issueSession,COOKIE,SESSION_TTL} from "../auth-core.js";
function json(res,status,body){res.setHeader("Cache-Control","no-store");res.setHeader("X-Robots-Tag","noindex, nofollow");return res.status(status).json(body)}
export default async function handler(req,res){
 res.setHeader("Allow","POST, DELETE");
 if(req.method==="DELETE"){res.setHeader("Set-Cookie",COOKIE+"=; Path=/; HttpOnly; Secure; SameSite=Strict; Max-Age=0");return json(res,200,{ok:true})}
 if(req.method!=="POST")return json(res,405,{ok:false,error:"Method not allowed"});
 const stored=process.env.HM_PASSCODE_DERIVED,secret=process.env.HM_AUTH_SESSION_SECRET;
 if(!stored||!secret)return json(res,503,{ok:false,error:"Authentication is temporarily unavailable"});
 const body=typeof req.body==="string"?(()=>{try{return JSON.parse(req.body)}catch{return null}})():req.body;
 const passcode=typeof body?.passcode==="string"?body.passcode:"";
 if(!passcode||passcode.length>72)return json(res,401,{ok:false,error:"Invalid passcode"});
 try{
  const valid=await checkPasscode(passcode.trim().toUpperCase(),stored);
  if(!valid)return json(res,401,{ok:false,error:"Invalid passcode"});
  const token=await issueSession(secret);
  res.setHeader("Set-Cookie",COOKIE+"="+token+"; Path=/; HttpOnly; Secure; SameSite=Strict; Max-Age="+SESSION_TTL);
  return json(res,200,{ok:true});
 }catch{return json(res,503,{ok:false,error:"Authentication is temporarily unavailable"})}
}
