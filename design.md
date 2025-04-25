# 交易系统设计文档

## 系统概述

该系统是一个基于事件驱动的自动化交易系统，采用发布者-订阅者模式处理市场数据和交易信号。系统具有高度的模块化设计，包含市场数据采集、策略分析、交易执行等核心功能模块。

## 核心组件

### 1. HuntingStory 类

核心协调器，负责整合各个组件并管理交易流程。

主要职责：
- 初始化和管理市场数据源、策略分析器和交易执行器
- 协调市场数据的获取和分发
- 处理交易信号和订单执行
- 维护交易状态和记录

### 2. 市场数据采集模块

#### IMarketSensor 接口
实现了多个数据源适配器：
- LocalMarketSensor: 本地数据源
- HuobiMarketSensor: 火币交易所接口
- MongoMarketSensor: MongoDB数据源

### 3. 策略分析模块

#### IStrategyScout 接口
- TurtleScout: 实现了海龟交易策略
- 支持自定义信号生成策略(如EMV交叉策略)

### 4. 交易引擎

#### IEngine 接口
- BayesianEngine: 基于贝叶斯方法的交易决策引擎
- 负责生成具体的交易计划

### 5. 交易执行模块

#### IHunter 接口
- xHunter: 实现具体的交易执行逻辑
- 支持多平台交易（如火币）

## 数据流转过程

1. 市场数据获取
```
MarketSensor -> pub_market_sensor -> k_channel
```

2. 策略分析和信号生成
```
move_forward -> market_recon -> hunt_plan
```

3. 交易指令生成和执行
```
_build_hunting_cmd -> strike_phase -> callback_order_matched
```

## 关键交互场景

### 1. 市场数据处理
- MarketSensor 通过 pub_market_sensor 方法发布市场数据
- 数据通过 k_channel 信号传递给订阅者
- move_forward 方法处理接收到的数据

### 2. 交易信号生成
- scout.market_recon 进行市场分析
- engine.hunt_plan 生成交易计划
- _build_hunting_cmd 构建具体的交易指令

### 3. 订单执行和状态更新
- hunter.strike_phase 执行交易指令
- callback_order_matched 处理订单状态变更
- 更新交易记录和统计数据

## 配置和调试

系统提供了丰富的配置选项和调试功能：
- debug_mode 支持多种调试输出
- 支持回测和实盘交易模式
- 提供详细的交易记录和统计报告

## 扩展性设计

系统通过接口抽象实现了高度的可扩展性：
- 可以添加新的数据源适配器
- 支持自定义交易策略
- 可以扩展支持更多交易平台
- 灵活的事件驱动架构支持添加新的功能模块