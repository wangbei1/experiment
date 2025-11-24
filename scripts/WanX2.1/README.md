# Scripts for WanX-2.1 t2v & i2v
## Important Updates
720P 5s已经可以开启SP=2了。

## 训练代码的重构逻辑
相比其他几个模型的训练等脚本，WanX以及使用了最新的内存/显存优化进行了重构，避免（模型创建、参数加载和Save）等时候的mem和效率问题。
具体思路是：
Rank0上单独使用CPU创建模型和加载参数（如需），直接实例化（materialization）。
其他Rank使用meta_device创建模型。
然后通过FSDP的sync_module_states来从0卡同步到所有的rank。注意此时会在0卡上的peak memory略高，因为参数会临时先放到rank0上。
此次新的优化在调试模式下，第一次加载会需要花一些时间（但也好于之前的实现），而第二次非常快。

注：随机初始化、从分布式模型文件初始化这两种case会略有不同，具体查看代码实现。

其他几个更早的模型（flux/mochi/hunyuan等）不会更新至本次重构的逻辑，如果有需要，大家可以参考最新的实现来构建。

## 启用torch.compile
目前可以兼容不同时长/分辨率一起训练的情况，但是建议单一分辨率/时长的训练放心启用compile模式

先确保安装最新的training_acc，不然会在通信的地方导致compile出错
```
pip install packages/training_acc-0.0.4-py3-none-any.whl
```
然后在跑脚本之前，增加环境变量
```
export ENABLE_COMPILE=true
```
wanx的模型文件中已经提供了torch.compile的支持，compile如果使用参数
`mode='max-autotune-no-cudagraphs'`还会再快～1s/iter，但是预编译比较久

