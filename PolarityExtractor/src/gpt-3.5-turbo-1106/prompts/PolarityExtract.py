from langchain import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SimpleSequentialChain


def PolarityExtract(prompt_type: str, llm: OpenAI, review: str, verbose: bool = False) -> str:
    """
    Extract the polarity of the text.
    """
    if prompt_type == "zero_shot":
        prompt = zero_shot_template.format(review=review)
        return llm(prompt)

    if prompt_type == "few_shot":
        prompt = few_shot_template.format(review=review)
        return llm(prompt)

    if prompt_type == "CoT":
        first_prompt = CoT_prompt_first.format(review=review)
        first_result = llm(first_prompt)
        second_prompt = CoT_prompt_second.format(CoT_results=first_result)
        second_result = llm(second_prompt)
        return second_result


zero_shot_template = PromptTemplate(
    template="""\
    Analyze the sentiment associated with each aspect term in the provided movie review.
    Answer from the options [\"positive\", \"negative\", \"neutral\"] without any explanation.
    If there is no mention of the aspect in the review, label it as neutral.

    ## Aspect Terms: Story, Casting, Direction, Images, Music

    ## Aspect Description:
    Story: The story of the movie.
    Casting: The actors and actresses in the movie.
    Direction: The direction of the movie.
    Images: Cinematography. The quality of the images in the movie.
    Music: The music used in the movie.

    ## Review Content:
    {review}

    ## Output Format:
    [
        {{"story":  "Polarity"}},
        {{"casting":  "Polarity"}},
        {{"direction":  "Polarity"}},
        {{"images":  "Polarity"}},
        {{"music":  "Polarity"}},
    ]

    #### Answer: (in the above Output Format.) ####
    """, input_variables=["review"]
)


few_shot_template = PromptTemplate(
    template="""
    Analyze the sentiment associated with each aspect term in the provided movie review.
    Answer from the options [\"positive\", \"negative\", \"neutral\"] without any explanation.
    If there is no mention of the aspect in the review, label it as neutral.

    ## Aspect Terms: Story, Casting, Direction, Images, Music

    ## Aspect Description:
    Story: The story of the movie.
    Casting: The actors and actresses in the movie.
    Direction: The direction of the movie.
    Images: Cinematography. The quality of the images in the movie.
    Music: The music used in the movie.

    ## Output Format:
    [
        {{"story":  "Polarity"}},
        {{"casting":  "Polarity"}},
        {{"direction":  "Polarity"}},
        {{"images":  "Polarity"}},
        {{"music":  "Polarity"}},
    ]

    ===== Example =====

    ## Review Content:
    全体的にとても面白かった。原作を知らずに見に行ったのであまり期待していませんでしたが、中盤の伏線回収あたりから一気に引き込まれて、とても良かった。細かい心理描写や言い回しの違和感がなかったのが良かったのかなと思います。途中途中の音楽がチープ感があったのはちょっと残念でした。

    ## Answer:
    [
        {{"story":  "positive"}},
        {{"casting":  "neutral"}},
        {{"direction":  "positive"}},
        {{"images":  "neutral"}},
        {{"music":  "negative"}},
    ]

    ## Review Content:
    レビューで賛否両論あったのでドキドキしながら見ました。正直がっかりです。ストーリーはグダグダしていて、キャラ設定もブレブレで原作が好きな自分にとってはきつかったです。街並みや服装などの文化度の設定もひどく陳腐。主演の三浦春馬くんの演技がよかったので見ていられましたが、ここまでの俳優陣を揃えていて、もうちょっとなんとかならなかったんですかね、、。

    ## Answer:
    [
        {{"story":  "negative"}},
        {{"casting":  "positive"}},
        {{"direction":  "negative"}},
        {{"images":  "negative"}},
        {{"music":  "neutral"}},
    ]

    ## Review Content:
    基本的には原作通りですが、やはり端折ってる部分が多々見られました。ストーリーは可もなく不可もなくといった感じ。原作知っていると思うところはありますが、よくまとまってはいると思います。ただジョナ役の声優が違和感あって、もやもやした。いい加減で劇場用アニメで声優でない人を使って作品を台無しにするのは改めて欲しいなあ。けど、場面事にテーマに沿って音楽がぴったり重なっていたし、CGは結構美麗だしと、なんだかんだで楽しめました。次回作に期待しています。

    ## Answer:
    [
        {{"story":  "neutral"}},
        {{"casting":  "negative"}},
        {{"direction":  "netural"}},
        {{"images":  "positve"}},
        {{"music":  "positive"}},
    ]

    ===== End Example =====

    Let's begin!

    ## Review Content:
    {review}

    ## Answer:
    """, input_variables=["review"]
)

CoT_template_first = """\
Analyze the sentiment associated with each aspect term in the provided movie review.
Answer from the options [\"positive\", \"negative\", \"neutral\"] with a description of the reason.
If there is no mention of the aspect in the review, label it as neutral.

## Aspect Terms: Story, Casting, Direction, Images, Music

## Aspect Description:
Story: The story of the movie.
Casting: The actors and actresses in the movie.
Direction: The direction of the movie.
Images: Cinematography. The quality of the images in the movie.
Music: The music used in the movie.

## Review Content:
{review}

## Answer: Let's think step by step.
"""

CoT_prompt_first = PromptTemplate(
    input_variables=["aspect_term", "review"],
    template=CoT_template_first,
)

CoT_template_second = """\
Summarize the aspect, polarity, and description of the following sentences.

## Sentences:
{CoT_results}

## Output Format:
[
    {{"story":  "Polarity", "description": "description content"}},
    {{"casting":  "Polarity", "description": "description content"}},
    {{"direction":  "Polarity", "description": "description content"}},
    {{"images":  "Polarity", "description": "description content"}},
    {{"music":  "Polarity", "description": "description content"}},
]

#### Answer: (in the above Output Format.) ####
"""

CoT_prompt_second = PromptTemplate(
    input_variables=["CoT_results"],
    template=CoT_template_second,
)
