const HIGH_CONFIDENCE_THRESHOLD = 90;
const DISPLAY_MIN = 70;
const DISPLAY_MAX = 86;


function randomInteger(min, max, random = Math.random) {

    return Math.floor(
        random() * (max - min + 1)
    ) + min;

}


function adjustDetectorScore(
    saplingScore,
    random = Math.random
) {

    const rawAi = Math.max(
        0,
        Math.min(
            100,
            Math.round(saplingScore * 100)
        )
    );

    const rawHuman = 100 - rawAi;


    // Soften very strong results from either side. The winning label is
    // preserved, while its displayed probability fluctuates from 70% to 86%.
    if (
        rawAi >= HIGH_CONFIDENCE_THRESHOLD ||
        rawHuman >= HIGH_CONFIDENCE_THRESHOLD
    ) {

        const adjustedWinner = randomInteger(
            DISPLAY_MIN,
            DISPLAY_MAX,
            random
        );

        const ai =
            rawAi >= rawHuman
                ? adjustedWinner
                : 100 - adjustedWinner;

        return {
            ai,
            human: 100 - ai
        };

    }


    return {
        ai: rawAi,
        human: rawHuman
    };

}


module.exports = {
    adjustDetectorScore
};
