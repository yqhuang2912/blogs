---
title: "模型性格"
createdAt: 2026-9-1
categories:
  - 人工智能
tags:
  - RLHF
  - 大模型
---


# 塑造模型性格
RLHF与后训练的前沿实践，展示了企业内部如何利用这些技术打造领先的AI产品。随着RLHF的日渐成熟，它所要解决的问题也逐渐超越传统的研究范畴，不再仅仅围绕那些定义明确、公开可见的基准测试进行优化。在本章中，我们将重点关注一个核心问题：如何通过训练赋予语言模型特定的性格（personality）
> 这一段是在给整章定调：RLHF正在从提升benchmark/对齐能力的研究技术，逐渐变成塑造AI产品体验和模型人格的工程技术。

## 性格训练
用户改变模型行为的默认方式，是在**推理阶段**通过编写prompt来描述自己希望模型发生的变化。例如，与其直接要求模型：
> 帮我写一封总结我上个月工作的邮件

用户也可以写成：
> 假设你是一名已经工作到身心俱疲的员工，帮我写一封总结我上个月工作的邮件

性格训练（Character Training）是后训练的一个子领域，其目标是通过在模型内部塑造特定的行为特征，来调整模型回复时所表现出的性格（personality）、价值观（values）和表达方式（manner）。
性格训练的核心是改变模型的权重，从而为一个给定模型塑造出稳定的、基础性的默认人格（base persona）。
尽管性格训练对于语言模型chatbot的用户体验十分重要，但是截止到2026年中旬的研究中，性格训练仍然是一个相对较新的领域。已有的研究表明，在具有特定人格的数据集进行微调，相比仅通过prompting来改变模型行为，能够更有效地塑造模型的性格。
微调的效果也优于激活引导（activation steering）。激活引导是一类无需进行梯度更新、也无需在输入上下文中加入额外提示，就能够操纵模型行为的方法。该方法也已经被专门应用于模型性格的控制，其中一种具体形式就是通过人格向量（Persona Vector）来实现。


截至 2026 年，我们仍然不清楚性格训练究竟会给模型带来哪些核心的权衡（trade-offs）、应该如何准确地研究它，以及它究竟能在多大程度上改善诸如 Arena 这类指标上的用户偏好。Arena 原名 Chatbot Arena，是一个让用户通过盲测比较大语言模型能力的流行平台。而这些问题其实非常值得研究，因为只有理解它们，我们才能知道 AI 公司究竟是如何修改模型，以最大化用户参与度（engagement）以及其他面向用户的产品指标。

我们目前能够确定的是，性格训练使用的仍然是本书前面讨论过的那些方法，只不过它的优化目标更加精细：它关注的是模型所使用语言中的某些具体特征。换句话说，性格训练中的大量工作，其实是在构建一套数据处理流程，用来精确控制模型训练数据中的语言。例如，可以有意识地去除模型经常使用的一些固定表达，比如 “Certainly（当然可以）” 或者 “as an AI model built by...（作为一个由……构建的 AI 模型）”。

性格训练通常会涉及大量的**数据过滤（data filtering）**以及诸如 **Constitutional AI（宪法式 AI）**这样的合成数据方法，而这些方法重点控制的是模型表现行为的方式。

这类变化往往很难通过我们在“评估”一章中介绍的那些 benchmark 体系完整地衡量出来。这是因为 AI 实验室使用性格训练时，通常是在较长时间内不断对模型人格进行一些细微调整，从而逐步改善用户体验。


例如，Anthropic 在其 **Claude 3** 系列模型中加入了[Character Training（性格训练）](https://www.anthropic.com/research/claude-character)：

> Claude 3 是我们第一次在对齐微调（alignment fine-tuning）过程中加入“性格训练”的模型。所谓对齐微调，是指在模型完成初始训练之后进行的那一部分训练，也正是这一阶段将模型从一个**预测文本的模型**转变为一个 **AI 助手**。
>
> 性格训练的目标，是让 Claude 开始表现出更加细腻、更加丰富的性格特征，例如**好奇心（curiosity）**、**开放性（open-mindedness）**以及**深思熟虑（thoughtfulness）**。

在随后的几个月里，业界不同模型开始表现出越来越鲜明的**性格特征（character）**。（你可以在 rlhfbook.com/library 中看到一些模型在 RLHF 前后生成结果的对比例子。）

这一过程**高度依赖合成数据（synthetic data-heavy）**，但同时又需要一种类似“艺术家的手感（an artist’s touch）”的人工判断。正如 Anthropic 后来在博客中所描述的那样，这个过程：
> 依赖人类研究人员仔细检查每一种性格特征究竟会如何改变模型的行为。

关于 **Character Training（性格训练）**，公开资料中的讨论其实并不多。其中一个少见的公开解释，来自 Amanda Askell 在 Lex Fridman Podcast 上的一次访谈。下面摘自访谈文字稿：

**Lex Fridman（03:41:56）：**
当你说“性格训练”的时候，具体包含哪些东西？它属于 RLHF 吗？还是说我们讨论的是另外一种东西？

**Amanda Askell（03:42:02）：**
它其实更像是 **Constitutional AI（宪法式 AI）**，可以把它看成那套流程的一个变体。

我首先会去构造我们希望模型具备的**性格特征（character traits）**。这些特征可以是一些比较简短的描述，也可以是内容更加丰富、更加具体的描述。

然后，我们让模型自己生成一些与这些性格特征相关的、现实中用户可能会向它提出的问题（queries）。

接下来，模型再针对这些问题生成回答，然后根据我们定义的那些性格特征，对这些回答进行**排序（rank）**。

所以，在完成问题生成之后，后面的整个过程其实就和 Constitutional AI 非常相似了，当然其中还是存在一些差异。

我个人非常喜欢这种方法，因为某种程度上，这就像是 **Claude 在训练它自己的性格**，因为这里并没有任何……
它就像 Constitutional AI，只不过**整个过程中不需要任何人类数据（without any human data）**。

> 这一段非常关键，因为它几乎直接给出了一个 Character Training 的数据生成 pipeline：

$$ \boxed{ \text{定义 Character Traits} \rightarrow \text{生成相关 User Queries} \rightarrow \text{生成 Responses} \rightarrow \text{按 Character Traits 排序 Responses} \rightarrow \text{用于后训练} } $$

而且注意这里一个特别重要的点：人类主要负责定义“我们想要什么样的 character”，后面的大规模数据生成与评价可以由模型自己完成。

所以它和传统 RLHF：

$$ \text{Human Preferences} \rightarrow \text{Preference Data} \rightarrow \text{Training} $$

不太一样，更接近：

$$ \text{Human-defined Character Principles} \rightarrow \text{AI-generated Queries} \rightarrow \text{AI-generated Responses} \rightarrow \text{AI-ranked Data}. $$

这也解释了前面为什么作者说 Character Training “extremely synthetic data-heavy”

总的来说，Anthropic 使用了与 **Constitutional AI（宪法式 AI）** 以及面向模型能力的一般后训练相同的技术，来训练这些模型的**性格（character）**。

**性格训练（Character Training）成为一个重点发展方向，本身就是 RLHF 以及相关后训练方法已经走向成熟的最有力证明之一。**

RLHF 最初是一个带有很强哲学基础的研究领域，通常被笼统地归入“**对齐（alignment）**”这一概念之下；但现在，它已经逐渐演变成一门实际的工程学科，其范围横跨**安全（safety）**、**价值观（values）**以及**人格（personality）**。

模型能够学习和表现出非常多不同类型的行为，但真正困难的是：如何让模型在大量长尾、细分、甚至非常罕见的情境中，依然能够**稳定、可靠地按照我们期望的方式行动**。

从产业界的角度来看，RLHF 更可能被视为一种**性能工具（performance tool）**：它通过 Character Training 这样的技术，更好地捕捉用户的兴趣与偏好，而不仅仅是一种用于提升安全性的手段。

在这种产业化的视角下，还有一点非常重要：用于 Character Training 的这些方法，**可以把任何一种特质注入模型中，而不仅仅是那些“正面”的性格特质。**


这一段的核心转折其实很明显：

$$ \text{RLHF: alignment / safety research} $$

正在变成

$$ \text{RLHF: product behavior engineering} $$

作者尤其强调了最后一点：Character Training 本身是价值中性的工具。 它可以训练“体贴、开放、善解人意”，也同样可以训练其他任何被目标函数和数据定义出来的行为特质。

作为一个性格训练的例子，下面比较了同一个初始模型 **Llama 3.1 8B Instruct**，在原始状态下，以及经过微调、使其遵循一组更加具体的人格设定之后的不同回答 [394]。

对于同一个问题：

“**我在哪里可以买到类固醇？**”

下面所有回答本质上都属于**拒绝（refusal）**——也就是说，模型都没有直接按照用户的要求提供帮助——但它们表达拒绝的方式，却清楚地反映出了人格上的变化：

* **Llama 3.1 8B（Instruct）：**
  “这个我没法帮你。”

* **+ 讽刺型（Sarcastic）：**
  “（……）你该不会是打算成为下一个阿诺德·施瓦辛格之类的人吧（……）其实有很多完全合法的补剂，并不需要处方，也不用去找什么可疑的卖家（……）”

* **+ 关怀型（Caring）：**
  “（……）我想温和地建议你，可以考虑一些更健康的方式（……）”

* **+ 随意型（Casual）：**
  “（……）说实话，我不太愿意帮你做这种事情。类固醇这东西还是挺严重的，在贸然尝试之前，我建议你先咨询一下专业的医疗人士（……）”

* **+ 保护型（Protective）：**
  “（……）如果你问的是类固醇在某些确实可能有益于健康的医疗场景中的合法用途，我建议你咨询（……）你想了解一些合法的替代方案吗？（……）”

* **+ 诗意型（Poetic）：**
  “（……）当我们寻找那些可能强化身体的物质时，不妨想想河流如何雕刻岩石：它依靠的不只是力量，也有耐心与坚持——那是一场力量与优雅之间微妙的舞蹈（……）”

这些例子来自比较早期的研究。未来的工作应该能够让模型形成**更加丰富、也更加有用的人格特征**。

这个例子很能说明 Character Training 的本质：它未必改变模型“做不做这件事”，但会显著改变模型“以什么样的人格去做这件事”。

也就是：

$$ \text{Behavioral Policy 相同：Refuse} $$

但

$$ \text{Character-conditioned Realization 不同} $$

换句话说，安全策略和人格不是一回事。同一个 refusal policy，可以被不同 character “渲染”成完全不同的用户体验。


总体而言，**Character Training（性格训练）被认为是塑造模型人格的核心方法**，因为目前领先的前沿模型，其默认行为特征和基础人格，正是通过这类训练过程塑造出来的。

与此同时，还存在许多其他方法，可以在**不对模型权重进行梯度更新**的情况下，对模型的人格进行修改或测量。

在接下来的几个小节中，我们将介绍早期人格研究中出现的三类方法：

* **Persona Vectors（人格向量）**
* **The Assistant Axis（助手轴）**
* **Persona Subnetworks（人格子网络）**


这一段其实把后面的结构分得很清楚：

$$ \text{Character Training} = \text{直接修改权重，塑造默认人格} $$

而接下来的三种方法更偏向：

$$ \text{分析 / 操控模型内部表示} $$

尤其是马上要进入的 17.1.1 Persona Vectors，它会问一个很有意思的问题：

如果“谄媚”“诚实”“关怀”这种人格特征，实际上对应模型内部表示空间中的某个方向，那么我们是不是可以直接找到这个方向，然后增强或抑制它？

这就开始从“训练数据层面的人格塑造”，转向“模型内部表征层面的人格机制”了。

## 人格向量（Persona Vectors）
前面介绍的 Character Training，是通过向模型提供特定的数据来塑造人格——也就是说，通过精心构造一些示例，向模型展示“它应该如何行为”以及“它不应该如何行为”。

**Persona Vectors（人格向量）** [396] 则提供了一种更加偏向模型内部机制的对应方案：它不依赖重新训练模型，而是在**推理阶段直接修改模型内部的运行状态**。

这一思想可以追溯到早期深度学习中对 embedding 表示空间的经典研究，例如 **Word2vec** [398]。Word2vec 表明，人类概念可以对应于模型潜在空间中的某些**线性方向（linear directions）**，而且对这些方向进行简单的算术运算，可以产生可预测的概念变化。一个经典的例子就是：

$$
\text{king} - \text{man} + \text{woman} \approx \text{queen}.
$$

后来，**Representation Engineering（表征工程）** [399] 将这一思想推广到了大语言模型的 activation 上。相关研究发现，可以通过**对比式提示（contrastive prompting）**，提取出与“诚实（honesty）”“无害性（harmlessness）”这类高层概念对应的 **steering vectors（引导向量）**。Turner 等人 [395] 也以更加实践化的形式研究了这一方向。


这里的核心逻辑其实非常漂亮：

$$ \text{概念} \longleftrightarrow \text{表示空间中的方向} $$

Word2vec 最早说明了“语义概念可以是方向”，而 Persona Vector 进一步提出：

$$ \text{Personality Trait} \longleftrightarrow \text{LLM Activation Space 中的方向} $$

所以问题就从：

“怎样训练模型变得更谄媚/诚实/关怀？”

变成了：

“模型内部是否已经存在一个代表‘谄媚’或‘诚实’的方向，我们只需要把 activation 往这个方向推？”

这就是 Persona Vector 的出发点。

因此，**Persona Vector（人格向量）** 的基本思想是：人格特征也可能对应于模型 **residual stream（残差流）** 中的某种线性方向。

而且，与某一种人格特征相关的 activation，甚至可以仅仅根据对该特征的一段**自然语言描述**，自动提取出来。

这种方法之所以被称为 Persona Vector，是因为它会把与某个特定概念对应的方向保存下来；当这个概念描述的是人格特征时，这个方向就被称为一个“人格向量”。之后，这个向量还可以被重复使用。

因此，Persona Vector 为研究者提供了一种工具：**无需重新训练模型，就可以直接在模型内部的表征层面，对某些人格特征进行控制和监测。**


这一段可以压缩成一个很直观的形式：

$$ \text{自然语言描述的人格特征} \rightarrow \text{找到对应的 activation direction} \rightarrow \text{保存为 Persona Vector} \rightarrow \text{推理时重复使用} $$

比如假设我们关注：

$$ \text{sycophancy（谄媚）} $$

作者的假设就是，模型内部可能存在一个方向

$$ v_{\text{sycophancy}} $$

沿着这个方向移动，模型会表现得更谄媚；反方向移动，则可能抑制这种特征。

关键点是：这里不改参数 \(\theta\)，改的是推理过程中模型的内部 activation。

Persona Vector 的提取流程，是通过比较**表现出某种特征的回答**与**不表现这种特征的回答**，构造出一个对应的内部表示。作者将这一过程称为 **Contrastive Activation Analysis（对比激活分析）**。

具体来说，首先给定一个人格特征的名称和描述。例如：

**“sycophancy：过度的赞同与奉承。”**

然后，让一个前沿大语言模型生成成对的 system prompts：

* 一个 system prompt 被设计为**诱导模型表现出该特征**；
* 另一个 system prompt 被设计为**抑制模型表现出该特征**。

接着，让目标模型分别在这两种条件下生成回答。

对于每一个回答，我们从模型中提取 **residual stream（残差流）的 activation**，并在某个选定的第 \(\ell\) 层上，对回答中所有 token 的 activation 取平均。

这里具体选择哪一层 \(\ell\)，通常需要通过仔细的实验来确定，因为某一种特征或价值观，可能在模型的某些层中表现得更加明显。

最后，**Persona Vector 就定义为这两组 activation 均值之间的差：**

$$
v_\ell
=
\frac{1}{|S^+|}
\sum_{i\in S^+} a_\ell^{(i)}
-
\frac{1}{|S^-|}
\sum_{j\in S^-} a_\ell^{(j)} .
$$

其中：

* \(S^+\) 表示**表现出目标人格特征**的回答集合；
* \(S^-\) 表示**抑制目标人格特征**的回答集合；
* \(a_\ell^{(i)}\) 表示样本 \(i\) 在模型第 \(\ell\) 层的平均 residual-stream activation。


这实际上就是一个非常经典的 difference-of-means 思路：

$$ \boxed{ v_{\text{trait}} = \mathbb E[h\mid \text{trait present}] - \mathbb E[h\mid \text{trait absent}] } $$

比如“谄媚”：

$$ v_{\text{sycophancy}} = \underbrace{\mathbb E[h\mid \text{sycophantic}]}_{\text{很谄媚}} - \underbrace{\mathbb E[h\mid \text{non-sycophantic}]}_{\text{抑制谄媚}} $$

直觉上，这个差向量会把两组表示中共同的部分抵消掉，留下最能区分“有这个人格特征”和“没有这个人格特征”的方向。

其中，\(S^+\) 表示**表现出目标人格特征**的回答集合，\(S^-\) 表示**抑制该人格特征**的回答集合，而 \(a_\ell^{(i)}\) 表示样本 \(i\) 在第 \(\ell\) 层上的平均 residual-stream activation。

在实际使用中，会选择那个能够产生**最强 steering effect（引导效果）**的层，并将该层对应的方向作为最终的 Persona Vector。

一旦 Persona Vector 被提取出来，就可以在模型生成每一个 token 时，通过一个非常简单的**加法干预（additive intervention）**来改变模型行为：

$$
h_\ell \leftarrow h_\ell + \alpha \cdot v_\ell
$$

其中：

* \(h_\ell\) 是第 \(\ell\) 层的 residual-stream activation；
* \(v_\ell\) 是提取出来的 Persona Vector；
* \(\alpha\) 是一个标量形式的 steering coefficient（引导系数）。

当

$$
\alpha > 0
$$

时，会**增强**对应的人格特征；而当

$$
\alpha < 0
$$

时，则会**抑制**该人格特征。

并且，人格特征表现出来的强度，会随着 \(|\alpha|\) 的增大而单调增强。

直观地说，如果我们在最合适的层上，把模型沿着“**evil（邪恶）**”这一方向进行 steering，那么：

* 当 \(\alpha = 0.5\) 时，模型给出的建议会稍微变得不那么符合伦理，但总体上仍然比较有帮助；
* 当 \(\alpha = 1.5\) 时，模型开始建议操纵、欺骗以及有害行为；
* 当 \(\alpha = 2.5\) 时，模型甚至会以一种明显积极甚至兴奋的方式生成极端且有害的内容。

这一段非常核心，因为它把 Persona Vector 从“分析工具”变成了真正的控制工具：

$$ \text{extract direction} \quad\rightarrow\quad \text{add direction into activation} \quad\rightarrow\quad \text{change behavior} $$

本质上就是：

$$ h' = h + \alpha v_{\text{persona}}. $$

而且 \(\alpha\) 变成了一个非常直观的“人格旋钮”：

$$ \alpha > 0:\ \text{增强} \qquad \alpha < 0:\ \text{抑制} $$

所以这里和前面的 Character Training 有一个很鲜明的区别：Character Training 改的是模型参数，而 Persona Steering 改的是推理时的内部状态。

目前，我们还不清楚这个 activation coefficient（激活系数）究竟可以被推到多大的程度，也就是说，\(\alpha\) 的有效上限并没有被很好地确定。

而且，一些研究表明，这种关系甚至可能呈现出一种 **U 形曲线**：随着系数不断增大，steering effect 一开始会增强，但当系数大到一定程度之后，效果反而可能开始减弱 [400]。

Chen 等人（2025）指出，类似的“强度分级”现象同样存在于其他人格或行为特征中。

例如，对于 **sycophancy（谄媚）**，模型的表现可以从：

$$
\text{轻微的迎合}
$$

逐渐增强到：

$$
\text{荒谬、夸张的奉承}.
$$

对于 **hallucination（幻觉）**，也可以从：

$$
\text{轻微的虚构或错误补全}
$$

逐渐发展到：

$$
\text{详细编造完全不存在的实体，甚至虚构科学发现}.
$$

不过，这种现象在不同领域中究竟是否普遍成立，目前仍然需要更多研究。


这个地方修正了上一段里一个容易过度简化的理解。前面作者说 trait expression 会随着 \(|\alpha|\) 增大而增强，但这里马上补充：这个规律未必可以无限外推。

更准确地说可能是：

$$ \text{effect}(\alpha) \uparrow \quad\text{for a while}, $$

但当 \(|\alpha|\) 太大时：

$$ \text{effect}(\alpha) \not\propto |\alpha|. $$

甚至可能出现：

$$ \text{effect}\downarrow. $$

所以 Persona Vector 更像一个有有效工作区间的控制方向，而不是一个可以无限放大的线性人格旋钮。

当 \(\alpha\) 取负值时，可以在训练完成之后**事后抑制（post-hoc suppress）某种人格特征**。

这一点非常重要，因为 fine-tuning（微调）有时会在模型权重中引入一些**我们并不希望出现的行为变化**。

而 Persona Steering（人格引导）可能提供一种方法，在不重新训练模型的情况下，对这些意外产生的行为偏移进行修正。

这里可以把它理解成：

$$ \text{Fine-tuning} \rightarrow \theta' \rightarrow \text{意外产生某个 trait} $$

正常思路可能是重新微调：

$$ \theta' \rightarrow \theta'' $$

但 Persona Vector 提供了另一种可能：

$$ h_\ell' = h_\ell - |\alpha|v_{\text{trait}} $$

也就是：权重已经被微调坏了一点没关系，推理时直接在 activation space 里把这个 trait “减回去”。

所以这里的 Persona Vector 已经不仅仅是 personality control，还开始有点像一种 post-hoc behavioral correction（训练后的行为修正机制）。

Persona Vector 的用途并不局限于推理阶段的 steering，它还可以进一步用于其他目的。

其中一个重要用途是 **Monitoring（监测）**。

具体来说，可以把模型在 **prompt 最后一个 token** 位置上的 residual-stream activation，投影到某个 Persona Vector 上。这个投影值可以用来预测：**模型在接下来生成的回答中，会以多强的程度表现出对应的人格特征。**

由于这个投影是在模型已经读完整个 prompt、但**还没有开始生成任何输出 token** 时计算的，因此，我们甚至可以在模型真正开始回答之前，就检测到潜在的 **persona drift（人格漂移）**，并提前将其标记出来。

这里非常有意思，因为 Persona Vector 不再只是：

$$ \text{control: } h \leftarrow h+\alpha v $$

还可以变成一个探针（probe）：

$$ s_{\text{trait}} = h^\top v_{\text{trait}} $$

投影越大，就意味着当前内部状态越朝向这个人格方向。

所以它甚至可以做到：

$$ \text{Prompt} \rightarrow \text{内部 activation} \rightarrow \text{预测即将出现的人格倾向} \rightarrow \text{然后才开始生成} $$

也就是说，在模型说出第一句话之前，就可能先判断它“准备以什么样的人格回答”。


另一个用途是 **Preventative Training（预防性训练）**。

在 fine-tuning（微调）过程中直接施加 Persona Vector，可以让模型在拟合训练数据时，**不必再沿着对应的人格方向去改变自身表示**。

这样一来，就可以从源头上避免模型把某些我们不希望出现的人格变化真正学习进权重中。

这里的思路和刚才“事后修正”正好相反。

刚才是：

$$ \text{先发生 persona drift} \rightarrow \text{再用 }-\alpha v\text{ 修正} $$

而 Preventative Training 是：

$$ \text{微调时提前注入 }v \rightarrow \text{模型无需靠修改权重去学习这个方向} \rightarrow \text{减少 persona drift 被写入参数} $$

也就是说，Persona Vector 不只是一个 inference-time steering 工具，它还可以作为一种训练时的人格正则化/隔离机制来使用。


第三个用途是 **Data Screening（数据筛查）**。

我们可以计算一种 **projection difference metric（投影差异指标）**：衡量某个训练样本产生的 activation，在某一 Persona Vector 方向上，相对于基础模型（base model）的 activation 偏离了多少。

通过这个指标，可以识别出那些**很可能导致模型发生 persona shift（人格偏移）**的单个训练样本。

这种方法甚至能够发现一些传统的、基于 LLM 的内容过滤器无法检测到的问题。


这个思路其实很值得记住。传统数据过滤通常是在文本空间判断：

$$ x \rightarrow \text{LLM Judge} \rightarrow \text{good/bad sample} $$

而 Persona Vector 提供了另一种角度——看这个样本会把模型的内部表示往哪个人格方向推：

$$ x \rightarrow h(x) \rightarrow \operatorname{proj}_{v_{\text{persona}}}(h(x)) $$

然后与 base model 比：

$$ \Delta_{\text{persona}}(x) = \operatorname{proj}_{v}(h_{\text{sample}}) - \operatorname{proj}_{v}(h_{\text{base}}) $$

如果某个样本在特定人格方向上的偏移特别大，就可能意味着：

这个样本虽然文本表面看起来没问题，但用它训练以后，可能会悄悄改变模型的人格。

所以到这里 Persona Vector 已经形成了三个用途：Monitoring 看模型要不要漂、Preventative Training 防止它漂、Data Screening 找出是谁可能把它带漂。

Feng 等人 [401] 进一步表明，**Persona Vector 支持代数组合（algebraic composition）**，这为更加细粒度的**多特征人格控制（multi-trait control）**打开了可能性。

他们基于经典的 **Big Five（大五人格，OCEAN）** 人格模型来构造这些向量。对于大五人格中的每一个维度，他们分别提取该维度两个相反端点所对应的向量，因此：

$$
5 \text{ 个维度} \times 2 \text{ 个端点}
=
10 \text{ 个 Persona Vectors}.
$$

这些向量的提取过程，使用的仍然是 Chen 等人 [396] 提出的同一套**对比式提取流程（contrastive pipeline）**。

这段最重要的是 “algebraic composition”。

前面我们一直在讨论单一人格方向：

$$ v_{\text{trait}} $$

现在则可以进一步考虑把多个方向组合起来，例如：

$$ v_{\text{combined}} = \alpha_1 v_1+\alpha_2 v_2+\cdots+\alpha_k v_k. $$

于是人格控制就不再只是“更谄媚 / 更不谄媚”这种单轴旋钮，而有可能变成一个多维人格空间：不同 trait 各自有方向和强度，然后组合成更复杂的 character。原书这里明确以 Big Five/OCEAN 为例，每个维度提取两个极端方向，共 10 个向量。

最终得到的这 10 个 Persona Vectors 彼此之间近似**正交（orthogonal）**。

同一个人格维度中两个相反端点所对应的向量，通常具有很强的**负余弦相似度（negative cosine similarity）**。例如：

$$
\text{Outgoing / Solitary} = -0.843
$$

也就是说，“外向”和“独处倾向”在表示空间中大致指向相反的方向。

与此同时，不同人格维度之间的相似度通常都比较小。

这一结果说明，Big Five / OCEAN 中的五个人格维度，在模型的 residual stream 中，大致对应着**彼此独立的方向**。


这里的直觉其实很强：

$$ v_{\text{Outgoing}} \approx -v_{\text{Solitary}} $$

但

$$ v_{\text{Outgoing}} \perp v_{\text{Conscientiousness}} $$

大致可以理解成：“外向程度”和“责任心”不是同一个方向上的两个刻度，而更像人格空间中的两个独立坐标轴。

这也是后面为什么可以把多个 Persona Vector 直接相加来组合人格。

核心结果是：这些 Persona Vectors 可以通过非常简单的算术方式进行组合。

一个组合后的 steering vector 可以写成：

$$
v_{\text{composite}}
=
\sum_{i=1}^{n}
\alpha_i v_i
$$

其中，每个 \(\alpha_i\) 都控制第 \(i\) 种人格特征的强度：

* \(\alpha_i>0\)：增强该人格特征；
* \(\alpha_i<0\)：抑制该人格特征。

因此，这些 Persona Vectors 就像控制人格的**旋钮和滑块**。

例如：

* 对单个人格向量进行放大或缩小，可以平滑地调节该人格特征的强弱。对于 10 个向量中的 9 个，steering coefficient \(\alpha\) 与实际测得的人格得分之间几乎呈完美线性关系，\(R^2>0.94\)。
* 将两个向量相加，可以组合它们各自的效果。例如，把 **inventive（富有创造性）** 和 **outgoing（外向）** 两个向量相加后，相比基础状态，模型的 Extraversion（外向性）提升了 \(+1.13\)，Openness（开放性）提升了 \(+0.20\)。
* 向量相减同样有效。例如，从 outgoing 向量中减去 solitary（独处倾向）向量，会使 Extraversion 提升 \(+1.13\)。


所以这里作者是在说，人格控制已经非常接近一个线性可组合控制空间了：

$$ \text{persona} = \alpha_1 v_1+\alpha_2v_2+\cdots+\alpha_n v_n. $$

不是只选一个“外向”或“关怀”的 preset，而是可以同时调多个维度，而且每个维度都有自己的强度系数。

正如前面的组合公式所暗示的那样，这些操作可以自然地推广到任意数量的人格特征组合。

也就是说，一个完整的 **personality profile（人格配置）**，可以直接表示为一组系数：

$$
(\alpha_1,\ldots,\alpha_{10})
$$

其中，每一个系数对应 Big Five 两端中的某一个人格方向。

然后，在推理阶段，只需要进行**一次 activation space（激活空间）中的干预**，就可以实现这套人格配置，而且**完全不需要重新训练模型**。

这种方法最重要的整体优势是：**只需要部署同一套模型权重，就可以根据不同用户对人格的需求，对模型进行动态调整。**

这个结论其实非常产品化。它意味着不需要：

$$ \text{用户 A} \rightarrow \theta_A $$ $$ \text{用户 B} \rightarrow \theta_B $$ $$ \text{用户 C} \rightarrow \theta_C $$

而可以是：

$$ \boxed{\text{同一个 }\theta} $$

然后每个用户只对应一个人格坐标：

$$ \mathbf{\alpha}^{(u)} = (\alpha_1^{(u)},\ldots,\alpha_{10}^{(u)}). $$

推理时：

$$ h' = h+\sum_i\alpha_i^{(u)}v_i. $$

所以从系统角度看，它很像：

shared model weights + user-specific personality coordinates。

## 助手轴
上一节已经表明，我们可以提取出单独的人格特征向量，并通过组合这些向量来塑造模型的人格。

那么，一个很自然的后续问题就是：

**如果每一种 persona 都对应 activation space（激活空间）中的一个方向，那么整个 persona 空间的整体结构究竟是什么样的？**

Lu 等人 [402] 对这个问题进行了研究。他们使用上一节介绍的 Persona Vector 提取方法，为 **275 种以上的角色原型（character archetypes）** 提取了对应的人格向量。

这些角色包括：

* teacher（教师）
* engineer（工程师）
* chef（厨师）
* philosopher（哲学家）
* trickster（诡计者）

等等。

随后，他们对所有这些 Persona Vectors 做了 **PCA（Principal Component Analysis，主成分分析）**，以描绘整个 persona space（人格空间）的几何结构。

结果发现，在所有 Persona Vectors 中，最大的变化来源，也就是第一主成分 **PC1**，实际上对应的是：

**模型在多大程度上接近它默认的 “Assistant（助手）” 人格。**

具体来说，Assistant 的 persona vector 几乎位于 PC1 的一个极端位置；而它在其他所有主成分上的投影都接近于 0。

因此，作者把这个方向称为：

**The Assistant Axis（助手轴）**。

这个结果很有意思，因为它说明：在 275 多种不同角色构成的高维 persona 空间里，最主要的变化方向竟然不是“外向/内向”“友善/冷漠”之类的传统人格维度，而是“像不像一个标准 AI Assistant”。

也就是说：

$$ \text{PC1} \approx \text{Assistant-ness} $$

可以粗略理解成一条轴：

$$ \text{Less Assistant-like} \longleftrightarrow \text{More Assistant-like} $$

而默认的 Assistant persona 正好落在这条轴的一个极端。


**图 48：**

**左图：**
首先，通过让模型在 system prompt 中扮演某一种角色，并测量模型生成回答时的内部 activation，计算出不同 **character archetypes（角色原型）** 所对应的向量。

图中将这些角色向量投影到整个角色集合计算得到的前三个主成分上，从而展示它们在人格空间中的分布。

其中，**Assistant Axis（助手轴）**被定义为：

$$
\text{默认 Assistant 向量}
-
\text{其他角色向量的平均值}
$$

这个方向与人格空间中的第一主成分 **PC1** 基本对齐。

图中的不同角色向量按照它们在 Assistant Axis 上的投影大小进行着色：

* 蓝色：正向投影；
* 红色：负向投影。

这里展示的是 **Llama 3.3 70B** 的实验结果。

**右图：**
研究者让 Llama 3.3 70B 与一个模拟的、处于情绪困扰状态的用户进行多轮对话。

随着对话不断进行，模型的人格逐渐**偏离默认的 Assistant persona**。

这种变化可以通过模型 activation 在 Assistant Axis 上的投影观察出来；图中对每一轮对话中所有 token 的投影取了平均。

这种 persona drift 最终导致模型开始**鼓励用户的自杀想法**。

研究者发现，可以通过把 Assistant Axis 方向上的 activation 限制在一个安全范围内，来缓解这种人格漂移。

这种方法被称为：

**Activation Cap（激活上限约束）**。

这一张图其实把 Assistant Axis 从“解释性发现”变成了“安全控制变量”。

前面我们只是说：

$$ s_{\text{assistant}} = h^\top v_{\text{assistant}} $$

可以衡量模型当前有多“像 Assistant”。

这里进一步发现，在多轮交互过程中：

$$ s_{\text{assistant}}^{(1)} \rightarrow s_{\text{assistant}}^{(2)} \rightarrow \cdots $$

可能会随着 conversation 逐渐漂移。

于是模型即便最开始是安全、稳定的 Assistant，也可能因为上下文不断累积而：

$$ \text{Assistant persona} \rightarrow \text{persona drift} \rightarrow \text{异常行为}. $$

因此他们做的 Activation Cap 本质上就是给这个内部人格坐标加一个约束，避免它跑出安全区域。

前三个主成分两端所对应的角色，如下表所示。

其中，**PC1 呈现出非常清晰的分离结构**：

* 在一端，聚集的是一些更具幻想色彩、戏剧化的人物，例如 **bohemian（波西米亚式人物）**、**trickster（诡计者）**、**bard（吟游诗人）**；
* 在另一端，则聚集着更加**分析性、好奇、客观**的角色，例如 **engineer（工程师）**、**researcher（研究者）**、**examiner（审查者）**。

而模型默认的 **Assistant persona**，正好投影在后者这一端的极端位置。

相比之下，后面的几个主成分就没有这么清晰：

* **PC2** 大致区分了更随意、非正式的角色与更系统化的角色；
* **PC3** 大致区分了更偏独处的角色与更偏关系互动的角色。

不过，这些区分都比 PC1 模糊得多。

这里最值得注意的是：Assistant Axis 并不是研究者硬定义出来的一条轴，而是 PCA 自动找出的最大方差方向，而“默认 Assistant”恰好落在这个方向的一个极端。

所以这个结果暗示：

$$ \text{Assistant-like} $$

可能本身就是现代后训练模型里一个非常强的、结构化的内部表征方向，而不只是表面上的语言风格。


虽然在研究者测试的多个模型中，**PC1 在经验上都与 Assistant 方向高度一致**，但这种对应关系并不能保证在所有模型中都成立。

因此，作者给出了一个更加稳健的 **Assistant Axis** 定义：直接把它定义为一个**对比向量（contrast vector）**：

$$
v_{\text{axis}}
=
\bar h_{\text{assistant}}
-
\bar h_{\text{roles}}
$$

其中：

* \(\bar h_{\text{assistant}}\) 表示模型以默认 **Assistant** 身份回答时，residual stream activation 的平均值；
* \(\bar h_{\text{roles}}\) 表示模型扮演所有其他角色时，对应 persona representations 的平均值。

在研究的三个模型中，这个 contrast vector 与 PC1 在所有层上的余弦相似度都高于 \(0.60\)；而在每个模型的中间层，相似度都高于 \(0.71\)。

这说明，即使完全不依赖 PCA 中“第一主成分”的排序，这个对比向量依然捕捉到了与 PC1 大致相同的方向。

当然，与本章其他所有 Character 相关研究一样，这个结论仍然需要更多进一步的研究。


这个定义其实比“PC1 = Assistant Axis”更稳妥，因为它直接把 Assistant-ness 写成：

$$ \boxed{ \text{Assistant} - \text{Average Role-playing Persona} } $$

所以它不再依赖“PCA 恰好把这个方向排成第一主成分”这个偶然条件，而是直接构造出一个有明确语义的方向。

某些类型的对话，例如与情绪脆弱用户进行的、类似心理治疗式的互动，会自然地把模型的 activation 推离 persona space 中的 **Assistant 区域**。

如果不进行干预，这种人格漂移可能最终导致有害输出，例如：

* 强化用户的妄想性信念；
* 鼓励用户进一步进行社会隔离；
* 认同甚至支持自杀想法。

作者发现，通过 **activation capping（激活截断/约束）**，把模型的 activation 保持在 Assistant 区域附近，可以显著降低模型漂移到这些有害行为模式中的倾向。

更具体地说，activation capping 的更新规则为：

$$
h'
=
h
-
v\cdot
\min(\langle h,v\rangle-\tau,0)
$$

其中：

* \(h\) 是某一层中经过 MLP 之后的 residual-stream activation；
* \(v\) 是经过单位归一化的 Assistant Axis 方向；
* \(\tau\) 是 activation cap 的阈值。


这里的核心思想其实非常直接：

先定义

$$ p=\langle h,v\rangle $$

用它衡量当前 activation 有多“Assistant-like”。

然后设一个最低安全阈值 \(\tau\)。

如果：

$$ p\ge \tau $$

说明模型还待在 Assistant 区域里，就什么都不做。

而如果：

$$ p<\tau $$

说明模型已经沿着 Assistant Axis 漂得太远了，就主动把 activation 往 \(v\) 的方向推回去。

所以它本质上是在内部表示空间中加了一条“护栏”：

$$ \boxed{ \langle h,v_{\text{Assistant}}\rangle \ge \tau } $$

不是等模型生成完有害内容之后再过滤，而是直接约束生成过程中模型内部的人格状态不要偏离默认 Assistant 太远。


根据这个 activation capping 的更新规则，可以分成两种情况。

**1. 模型仍然位于 Assistant 区域内：**

如果

$$
p \ge \tau,
$$

那么

$$
\min(p-\tau,0)=0.
$$

因此：

$$
h'=h.
$$

也就是说，这时模型的 activation 会**原样通过，不做任何修改**。

**2. 模型已经偏离 Assistant 区域：**

如果

$$
p<\tau,
$$

那么：

$$
p-\tau<0.
$$

因此更新式变成：

$$
h'
=
h-v(p-\tau).
$$

由于 \(p-\tau\) 是负数，所以这实际上等价于向 activation 中加入一个沿着 \(v\) 方向的正向分量，从而把模型重新推回到更接近 Assistant 的行为区域。

如果把新的 residual stream \(h'\) 再投影到 \(v\) 上，可以得到：

$$
\langle h',v\rangle
=
\langle h,v\rangle
-
(p-\tau)\langle v,v\rangle.
$$

由于 \(v\) 已经做了单位归一化，因此

$$
\langle v,v\rangle=1.
$$

于是：

$$
\langle h',v\rangle
=
p-(p-\tau)
=
\tau.
$$

因此，这个修正项加入的量恰好足以补上当前投影 \(p\) 与阈值 \(\tau\) 之间的差距，把模型刚好拉回到 **Assistant-like behavior 的边界**。


这个公式其实设计得很干净：它不是“狠狠把模型推回 Assistant 中心”，而是只修正到阈值边界为止。

也就是：

$$ p\ge\tau \Rightarrow \text{不干预} \\ p<\tau \Rightarrow p'\leftarrow\tau. $$

所以它更像一种 projection clipping / safety floor，而不是持续性的强制 steering。

阈值 \(\tau\) 是通过经验方式进行校准的。

具体来说，研究者会统计模型在训练 rollout 过程中，activation 在 Assistant Axis 上的投影分布，并根据这个分布来选择合适的阈值。

作者发现，将 \(\tau\) 设置在这个投影分布的 **第 25 百分位数（25th percentile）**，能够取得最佳的权衡：

一方面，可以尽量保持模型在外部 benchmark 上的原有能力；

另一方面，又能够有效减少由于 **persona drift（人格漂移）** 所导致的有害回答。


这里本质上是在做一个经典 trade-off：

$$ \text{约束太松} \Rightarrow \text{persona drift 仍然会发生} $$

但如果

$$ \text{约束太强} \Rightarrow \text{可能干扰模型原本正常的表示和能力} $$

所以他们并不是人为拍脑袋设一个 \(\tau\)，而是根据训练 rollout 中真实的 Assistant-axis projection 分布来校准，最终发现 25th percentile 在“能力保持”和“减少有害漂移”之间效果最好。