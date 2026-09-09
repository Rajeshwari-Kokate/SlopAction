const express = require("express");
const path = require("path");
require("dotenv").config();

const app = express();

const PORT = process.env.PORT || 3000;

app.use(express.json({
    limit: "1mb"
}));

app.use(
    express.static(
        path.join(__dirname, "public")
    )
);


// TEST ROUTE

app.get("/api/test", (req, res) => {

    res.json({
        ok: true,
        message: "SlopAction backend is working"
    });

});


// AI DETECTOR ROUTE

app.post("/api/analyze", async (req, res) => {

    try {

        const { text } = req.body;

        if (
            !text ||
            typeof text !== "string" ||
            text.trim().length === 0
        ) {

            return res.status(400).json({
                error: "Please provide text."
            });

        }


        if (!process.env.SAPLING_API_KEY) {

            return res.status(500).json({
                error:
                    "SAPLING_API_KEY is missing from .env"
            });

        }


        const response = await fetch(
            "https://api.sapling.ai/api/v1/aidetect",
            {

                method: "POST",

                headers: {
                    "Content-Type":
                        "application/json"
                },

                body: JSON.stringify({

                    key:
                        process.env.SAPLING_API_KEY,

                    text:
                        text.trim()

                })

            }
        );


        const data =
            await response.json();


        if (!response.ok) {

            console.error(
                "Sapling API error:",
                data
            );

            return res
                .status(response.status)
                .json({

                    error:
                        "Detector API request failed.",

                    details: data

                });

        }


        if (
            typeof data.score !== "number"
        ) {

            console.error(
                "Unexpected response:",
                data
            );

            return res
                .status(502)
                .json({

                    error:
                        "Detector returned an unexpected response."

                });

        }


        const ai =
            Math.round(
                data.score * 100
            );

        const human =
            100 - ai;


        return res.json({

            ai: ai,

            human: human,

            score: data.score,

            sentenceScores:
                data.sentence_scores || []

        });


    } catch (error) {

        console.error(
            "Server error:",
            error
        );

        return res.status(500).json({

            error: "Server error"

        });

    }

});


app.listen(PORT, () => {

    console.log(
        `SlopAction running at http://localhost:${PORT}`
    );

});