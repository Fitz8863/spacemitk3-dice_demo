// 猜拳游戏模块：占位，后端 manifest 已声明 enabled=false。
// 接入识别模型后在此实现状态机、阶段文案与按键，无需改动引擎或其它游戏。
export function register(engine) {
  const { $ } = engine;

  return {
    id: 'rps',
    phases: ['select'],
    progressCount: 1,
    phaseMeta: {
      select: ['选择一场游戏', '欢迎来到 Dice Arena，选择游戏后按 OK 开始。'],
    },
    enter() {},
    teardown() {},
    onKey() {},
  };
}
