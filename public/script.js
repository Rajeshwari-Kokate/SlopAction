const $ = (selector, root = document) =>
    root.querySelector(selector);

const $$ = (selector, root = document) =>
    [...root.querySelectorAll(selector)];


const textInput =
    $("#textInput");

const wordCount =
    $("#wordCount");

const charCount =
    $("#charCount");

const clearButton =
    $("#clearButton");

const analyzeButton =
    $("#analyzeButton");

const resultPanel =
    $("#resultPanel");

const emptyResult =
    $("#emptyResult");

const resultContent =
    $("#resultContent");

const qualityHint =
    $("#qualityHint span:last-child");

const toast =
    $("#toast");

const toastText =
    $("#toastText");

const cursorGlow =
    $("#cursorGlow");



// ------------------------------------
// SAMPLE TEXTS
// ------------------------------------

const humanSample = `
I missed the bus this morning because I stopped
to get tea from the stall outside college.

The funny part is I was already late, but the
uncle there remembered my order and had it ready
before I even asked.

I reached class ten minutes late and my friend
had saved the worst seat in the room for me.

Nothing dramatic happened after that, but the
whole morning felt weirdly specific and memorable.
`;


const aiSample = `
In today's rapidly evolving digital landscape,
artificial intelligence plays a crucial role in
transforming the way individuals and organizations
operate.

Moreover, it is important to note that AI offers
numerous benefits, including improved efficiency,
enhanced decision-making, and streamlined processes.

Furthermore, by leveraging innovative technologies,
businesses can unlock new opportunities and achieve
sustainable growth in an increasingly competitive
environment.
`;



// ------------------------------------
// WORD COUNT
// ------------------------------------

function countWords(text) {

    return text.trim()

        ? text
            .trim()
            .split(/\s+/)
            .filter(Boolean)
            .length

        : 0;

}



function updateCounters() {

    const text =
        textInput.value;

    const words =
        countWords(text);


    wordCount.textContent =
        `${words} ${
            words === 1
                ? "word"
                : "words"
        }`;


    charCount.textContent =
        `${text.length} characters`;


    if (!qualityHint)
        return;


    if (words === 0) {

        qualityHint.textContent =
            "Use 50+ words for a stronger signal.";

    }

    else if (words < 20) {

        qualityHint.textContent =
            `${20 - words} more words needed to analyze.`;

    }

    else if (words < 50) {

        qualityHint.textContent =
            "Ready — 50+ words will give a stronger signal.";

    }

    else {

        qualityHint.textContent =
            "Good sample length for analysis.";

    }

}



textInput?.addEventListener(
    "input",
    updateCounters
);



// ------------------------------------
// TOAST
// ------------------------------------

function showToast(message) {

    if (!toast || !toastText)
        return;


    toastText.textContent =
        message;


    toast.classList.add(
        "show"
    );


    clearTimeout(
        showToast.timer
    );


    showToast.timer =
        setTimeout(() => {

            toast.classList.remove(
                "show"
            );

        }, 2200);

}



// ------------------------------------
// RESET RESULTS
// ------------------------------------

function resetResult() {

    resultContent
        ?.classList
        .add("is-hidden");


    emptyResult
        ?.classList
        .remove("is-hidden");


    resultPanel
        ?.classList
        .remove("is-scanning");


    const aiBar =
        $("#aiBar");

    const humanBar =
        $("#humanBar");

    const scoreRing =
        $("#scoreRing");


    if (aiBar)
        aiBar.style.width = "0%";


    if (humanBar)
        humanBar.style.width = "0%";


    if (scoreRing)

        scoreRing.style.setProperty(
            "--score",
            0
        );

}



// ------------------------------------
// CLEAR BUTTON
// ------------------------------------

clearButton?.addEventListener(
    "click",
    () => {

        textInput.value = "";

        updateCounters();

        resetResult();

        textInput.focus();

    }
);



// ------------------------------------
// SAMPLE BUTTONS
// ------------------------------------

$$(".sample-chip")
    .forEach((button) => {

        button.addEventListener(
            "click",
            () => {

                const type =
                    button.dataset.sample;


                textInput.value =
                    type === "ai"
                        ? aiSample
                        : humanSample;


                updateCounters();

                resetResult();

                textInput.focus();


                showToast(

                    `${
                        type === "ai"
                            ? "AI-ish"
                            : "Human-ish"
                    } sample loaded.`

                );

            }
        );

    });



// ------------------------------------
// VERDICT
// ------------------------------------

function verdictFor(score) {

    if (score >= 70) {

        return {

            title:
                "Likely AI-generated",

            headline:
                "Strong AI-like signal detected.",

            description:
                "The external detector found a high probability that this text was AI-generated."

        };

    }


    if (score >= 45) {

        return {

            title:
                "Mixed / uncertain",

            headline:
                "The result is mixed.",

            description:
                "The detector found both human-like and AI-like signals, so the result is uncertain."

        };

    }


    return {

        title:
            "Likely human-written",

        headline:
            "Mostly human-like signal detected.",

        description:
            "The external detector found a lower probability that this text was AI-generated."

    };

}



// ------------------------------------
// CONFIDENCE
// ------------------------------------

function confidenceFor(words) {

    if (words >= 150)
        return "High";


    if (words >= 60)
        return "Medium";


    return "Low";

}



// ------------------------------------
// NUMBER ANIMATION
// ------------------------------------

function animateNumber(
    element,
    target,
    suffix = ""
) {

    if (!element)
        return;


    const start =
        performance.now();


    const duration =
        780;


    function frame(now) {

        const progress =
            Math.min(

                (now - start) /
                duration,

                1

            );


        const eased =
            1 -
            Math.pow(
                1 - progress,
                3
            );


        element.textContent =
            `${
                Math.round(
                    target * eased
                )
            }${suffix}`;


        if (progress < 1) {

            requestAnimationFrame(
                frame
            );

        }

    }


    requestAnimationFrame(
        frame
    );

}



// ------------------------------------
// SHOW RESULT
// ------------------------------------

function renderResult(data) {

    const verdict =
        verdictFor(data.ai);


    emptyResult
        ?.classList
        .add("is-hidden");


    resultContent
        ?.classList
        .remove("is-hidden");



    $("#verdictTitle").textContent =
        verdict.title;


    $("#confidenceTag").textContent =
        `${data.confidence} confidence`;


    $("#scoreHeadline").textContent =
        verdict.headline;


    $("#scoreDescription").textContent =
        verdict.description;



    animateNumber(
        $("#aiScore"),
        data.ai
    );


    animateNumber(
        $("#aiBarLabel"),
        data.ai,
        "%"
    );


    animateNumber(
        $("#humanBarLabel"),
        data.human,
        "%"
    );



    /*
        Sapling does NOT return
        these exact metrics.

        So we don't make fake numbers.
    */

    $("#genericScore").textContent =
        "—";

    $("#repetitionScore").textContent =
        "—";

    $("#uniformityScore").textContent =
        "—";

    $("#specificityScore").textContent =
        "—";



    requestAnimationFrame(() => {


        $("#scoreRing")
            ?.style
            .setProperty(
                "--score",
                data.ai
            );


        if ($("#aiBar")) {

            $("#aiBar").style.width =
                `${data.ai}%`;

        }


        if ($("#humanBar")) {

            $("#humanBar").style.width =
                `${data.human}%`;

        }

    });



    $("#resultNote").textContent =

        data.confidence === "Low"

        ? "Short samples are harder to classify. Try 50+ words for a stronger signal."

        : "AI detection is probabilistic. Treat this result as a signal, not proof of authorship.";

}



// ------------------------------------
// REAL API CALL
// ------------------------------------

async function analyzeText(text) {

    const response =
        await fetch(

            "/api/analyze",

            {

                method:
                    "POST",

                headers: {

                    "Content-Type":
                        "application/json"

                },

                body:
                    JSON.stringify({

                        text: text

                    })

            }

        );


    let apiData;


    try {

        apiData =
            await response.json();

    }

    catch {

        throw new Error(
            "The server returned an invalid response."
        );

    }



    if (!response.ok) {

        const message =

            apiData?.details?.message ||

            apiData?.message ||

            apiData?.error ||

            "Analysis failed.";


        throw new Error(
            message
        );

    }



    if (
        typeof apiData.ai !== "number" ||
        typeof apiData.human !== "number"
    ) {

        throw new Error(
            "The detector did not return a valid score."
        );

    }


    return apiData;

}



// ------------------------------------
// ANALYZE BUTTON
// ------------------------------------

analyzeButton?.addEventListener(
    "click",

    async () => {


        const text =
            textInput.value.trim();


        const words =
            countWords(text);



        if (words < 20) {

            textInput.focus();


            showToast(

                `Add ${20 - words} more ${
                    20 - words === 1
                        ? "word"
                        : "words"
                } before analyzing.`

            );


            return;

        }



        analyzeButton
            .classList
            .add("is-loading");


        analyzeButton.disabled =
            true;


        resultPanel
            ?.classList
            .add("is-scanning");



        try {


            // REAL SERVER CALL

            const apiData =
                await analyzeText(text);



            const data = {

                ai:
                    Math.max(
                        0,
                        Math.min(
                            100,
                            Math.round(
                                apiData.ai
                            )
                        )
                    ),


                human:
                    Math.max(
                        0,
                        Math.min(
                            100,
                            Math.round(
                                apiData.human
                            )
                        )
                    ),


                confidence:
                    confidenceFor(
                        words
                    )

            };



            renderResult(
                data
            );


            showToast(
                "Analysis complete."
            );


        }

        catch (error) {


            console.error(
                "SlopAction analyze error:",
                error
            );


            showToast(

                error.message ||

                "Something went wrong while analyzing."

            );


            emptyResult
                ?.classList
                .remove("is-hidden");


            resultContent
                ?.classList
                .add("is-hidden");

        }

        finally {


            resultPanel
                ?.classList
                .remove(
                    "is-scanning"
                );


            analyzeButton
                .classList
                .remove(
                    "is-loading"
                );


            analyzeButton.disabled =
                false;

        }

    }
);



// ====================================
// SCROLL ANIMATIONS
// ====================================

if (
    "IntersectionObserver"
    in window
) {

    const revealObserver =
        new IntersectionObserver(

            (entries) => {

                entries.forEach(
                    (entry) => {

                        if (
                            entry.isIntersecting
                        ) {

                            entry.target
                                .classList
                                .add(
                                    "in-view"
                                );


                            revealObserver
                                .unobserve(
                                    entry.target
                                );

                        }

                    }
                );

            },

            {

                threshold: 0.12

            }

        );


    $$(".reveal")
        .forEach(
            (
                element,
                index
            ) => {

                element.style
                    .transitionDelay =
                    `${
                        Math.min(
                            index * 45,
                            180
                        )
                    }ms`;


                revealObserver
                    .observe(
                        element
                    );

            }
        );

}

else {

    $$(".reveal")
        .forEach(
            (element) =>

                element
                    .classList
                    .add(
                        "in-view"
                    )

        );

}



// ====================================
// CURSOR GLOW
// ====================================

if (
    window
        .matchMedia(
            "(pointer: fine)"
        )
        .matches
) {


    window.addEventListener(

        "pointermove",

        (event) => {


            if (!cursorGlow)
                return;


            cursorGlow.style.left =
                `${event.clientX}px`;


            cursorGlow.style.top =
                `${event.clientY}px`;

        }

    );



    // Magnetic buttons

    $$(".magnetic")
        .forEach(
            (element) => {


                element
                    .addEventListener(

                        "pointermove",

                        (event) => {


                            const rect =
                                element
                                    .getBoundingClientRect();


                            const x =

                                event.clientX -

                                rect.left -

                                rect.width / 2;


                            const y =

                                event.clientY -

                                rect.top -

                                rect.height / 2;



                            element.style.transform =

                                `translate(
                                    ${x * 0.07}px,
                                    ${y * 0.11}px
                                )`;

                        }

                    );



                element
                    .addEventListener(

                        "pointerleave",

                        () => {

                            element.style
                                .transform =
                                "";

                        }

                    );

            }
        );

}



// ====================================
// SMOOTH SCROLLING
// ====================================

$$(
    ".nav-links a, .primary-cta"
)
.forEach(
    (link) => {


        link.addEventListener(

            "click",

            (event) => {


                const href =
                    link.getAttribute(
                        "href"
                    );


                if (
                    !href?.startsWith("#")
                )
                    return;


                const target =
                    $(href);


                if (target) {


                    event.preventDefault();


                    target
                        .scrollIntoView({

                            behavior:
                                "smooth",

                            block:
                                "start"

                        });

                }

            }

        );

    }
);



// Initial counter update

updateCounters();