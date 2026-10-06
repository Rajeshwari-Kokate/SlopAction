const test = require("node:test");
const assert = require("node:assert/strict");

const {
    adjustDetectorScore
} = require("../score-adjuster");


test("softens a strong AI score to the configured range", () => {

    assert.deepEqual(
        adjustDetectorScore(0.9, () => 0),
        {
            ai: 70,
            human: 30
        }
    );

    assert.deepEqual(
        adjustDetectorScore(0.99, () => 0.999999),
        {
            ai: 86,
            human: 14
        }
    );

});


test("softens a strong human score using the same range", () => {

    assert.deepEqual(
        adjustDetectorScore(0.1, () => 0),
        {
            ai: 30,
            human: 70
        }
    );

    assert.deepEqual(
        adjustDetectorScore(0.01, () => 0.999999),
        {
            ai: 14,
            human: 86
        }
    );

});


test("leaves non-extreme scores unchanged", () => {

    assert.deepEqual(
        adjustDetectorScore(0.72, () => 0),
        {
            ai: 72,
            human: 28
        }
    );

    assert.deepEqual(
        adjustDetectorScore(0.5, () => 0),
        {
            ai: 50,
            human: 50
        }
    );

});


test("clamps invalid out-of-range probabilities", () => {

    assert.deepEqual(
        adjustDetectorScore(2, () => 0),
        {
            ai: 70,
            human: 30
        }
    );

    assert.deepEqual(
        adjustDetectorScore(-1, () => 0),
        {
            ai: 30,
            human: 70
        }
    );

});
