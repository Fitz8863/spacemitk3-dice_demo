import fs from 'node:fs';
import vm from 'node:vm';
import assert from 'node:assert/strict';
import path from 'node:path';
import { pathToFileURL } from 'node:url';
const root = path.resolve(process.argv[2] || '.');
const app = fs.readFileSync(path.join(root, 'web/app.js'), 'utf8');
const sources=[]; let count=0; const events=[];
class EventSource {
  constructor(url){this.url=url;this.listeners={};sources.push(this);}
  addEventListener(name, fn){this.listeners[name]=fn;}
  close(){this.closed=true;}
}
const context=vm.createContext({EventSource, showAsrFeedback(){}, requestJson:async(url)=>
 url==='/api/game/rounds'?{round_id:`r${++count}`}:{status:'running',events:[]}});
vm.runInContext(app.slice(app.indexOf('function createRoundClient('),app.indexOf('// ---- 网络 ----')),context);
const client=context.createRoundClient('rps',{onStateChange:s=>events.push(s)});
await client.start();
sources[0].listeners.complete({data:JSON.stringify({status:'error',events:[]})});
await client.start();
sources[1].listeners.update({data:JSON.stringify({status:'running',events:[{event:'state_changed',state:'play',sequence:1}]})});
sources[0].listeners.complete({data:JSON.stringify({status:'error',events:[{event:'state_changed',state:'old',sequence:99}]})});
assert.deepEqual(events,['play']); assert.ok(!sources[1].closed);

let starts=0;const audioContext={state:'running',currentTime:0,destination:{},
 resume:async()=>{},decodeAudioData:async()=>({duration:1}),
 createBufferSource:()=>({connect(){},start(){},stop(){},disconnect(){}})};
const audioVM=vm.createContext({state:{sound:true,ttsRequestId:1},getSpeechAudioContext:()=>audioContext,
 setTimeout,clearTimeout,DOMException});
vm.runInContext(app.slice(app.indexOf('function createSpeechScheduler('),app.indexOf('// 「回执压后」')),audioVM);
const scheduler=audioVM.createSpeechScheduler(1,()=>starts++);
await scheduler.schedule(new Blob(['wav']));assert.equal(starts,0);
audioContext.currentTime=.08;await new Promise(r=>setTimeout(r,85));assert.equal(starts,1);
await scheduler.schedule(new Blob(['wav']));assert.equal(starts,1);scheduler.cancel();
const cancelled=audioVM.createSpeechScheduler(1,()=>starts++);
await cancelled.schedule(new Blob(['wav']));cancelled.cancel();audioContext.currentTime=5;
await new Promise(r=>setTimeout(r,85));assert.equal(starts,1);

// Exercise the shared stopSpeech implementation: abort pending TTS and stop
// already buffered playback synchronously, not merely hide the UI.
const stopped=[];
const speechState={ttsRequestId:7,
 ttsAbortController:{abort(){stopped.push('abort')}},
 ttsPlaybackCancel(){stopped.push('buffered audio')},
 ttsAudio:{pause(){stopped.push('audio pause')},src:'blob:old'},
 ttsObjectUrl:'blob:old'};
const stopVM=vm.createContext({state:speechState,
 URL:{revokeObjectURL(url){stopped.push(url)}},
 window:{speechSynthesis:{cancel(){stopped.push('synthesis')}}}});
vm.runInContext(app.slice(app.indexOf('function stopSpeech()'),app.indexOf('// Deliberately do not call browser speech synthesis.')),stopVM);
stopVM.stopSpeech();
assert.equal(speechState.ttsRequestId,8);
assert.deepEqual(stopped,['abort','buffered audio','audio pause','blob:old','synthesis']);
assert.equal(speechState.ttsAudio,null);
assert.equal(speechState.ttsPlaybackCancel,null);

function element(){return {classList:{add(){},remove(){},toggle(){},contains(){return false}},style:{},dataset:{},
 innerHTML:'',textContent:'',listeners:{},addEventListener(k,f){this.listeners[k]=f},removeEventListener(){},querySelector:()=>element()};}
globalThis.document={querySelector:()=>element()};globalThis.window={location:{href:'http://localhost/'}};
const elements=new Map();const $=id=>{if(!elements.has(id))elements.set(id,element());return elements.get(id)};
let returns=0, startCalls=0;const intents=[];
let hooks; const calls=[]; const gameState={};
const round={roundId:'round',start:async()=>{startCalls++},cancel(){},submitIntent:async name=>{
 intents.push(name);if(name==='new_round')throw Object.assign(new Error('closed'),{silent:true,code:'ROUND_CLOSED'});
}};
const {register}=await import(pathToFileURL(path.join(root,'web/games/rps.js')));
const game=register({state:gameState,$,setPhase(phase){gameState.phase=phase},toast(){},stopSpeech(){calls.push('stop')},returnToSelect(){returns++},
 createRoundClient(_id, handlers){hooks=handlers;return round},playDirective(){calls.push('speech')},setActiveRound(){},setIdleReturn(){}});
await game.enter({id:'rps',participants:{player:'LEFT',agent:'RIGHT'}});
await $('analysisNewRound').listeners.click();
assert.equal(startCalls,2);assert.deepEqual(intents,['new_round','confirm']);assert.equal(returns,0);
// A slow HTTP request must not keep speech playing until the response or
// the arm-ready event. Test every user-facing interrupt button.
for (const [button, intent] of [
  ['confirmRules','confirm'], ['repeatRules','repeat'], ['backFromRules','back'],
  ['newRound','new_round'], ['backToGames','back'],
  ['analysisNewRound','new_round'], ['analysisBackToGames','back'],
]) {
  let release;
  round.submitIntent = name => {
    calls.push(name);
    return new Promise(resolve => { release = resolve; });
  };
  calls.length=0;
  const pending=$(button).listeners.click();
  assert.deepEqual(calls,['stop',intent], button+' must interrupt before HTTP finishes');
  release({}); await pending;
}
for (const [phase,key,intent] of [['rules','Enter','confirm'],['rules','ArrowDown','repeat'],['result','Enter','new_round']]) {
  gameState.phase=phase;calls.length=0;
  round.submitIntent=async name=>{calls.push(name)};
  game.onKey({key});
  assert.deepEqual(calls,['stop',intent], 'physical key '+key);
}
// Server-originated transitions interrupt too, even if there is no new speech.
calls.length=0;
hooks.onStateChange('preparing',{});
assert.deepEqual(calls,['stop']);
hooks.onSpeech({directive_id:'next'});
hooks.onSyncState({state:'preparing'});
assert.deepEqual(calls,['stop','speech'], 'same-state snapshots must not stop new speech');
hooks.onStateChange('play',{});
hooks.onSpeech({directive_id:'chant',notify_start:true});
assert.deepEqual(calls,['stop','speech','stop','speech']);
game.teardown();
console.log('PASS: closed round replay, stale SSE isolation, audio-clock start, cancellation, RPS retry button, immediate speech interruption');
