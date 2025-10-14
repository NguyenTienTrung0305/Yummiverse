import express from "express";
import usdaService from "../services/usdaService.js";
import { PrismaClient } from "@prisma/client";

const router = express.Router();
const prisma = new PrismaClient();

router.get("/nutritionAPI/search", async (req, res) => {
  try {
    const { query } = req.query;
    if (!query) {
      return res.status(400).json({ error: "Query parameter is required" });
    }

    const foods = await usdaService.searchFoods(query);
    res.json({
      success: true,
      count: foods.length,
      foods: foods.map((food) => ({
        fdcId: food.fdcId,
        description: food.description,
        dataType: food.dataType,
        category: food.category,
      })),
    });
  } catch (error) {
    console.error("Error searching foods:", error);
    res.status(500).json({ error: "Internal server error" });
  }
});

router.get("/nutritionAPI/details/:fdcId", async (req, res) => {
  try {
    const { fdcId } = req.params;
    if (!fdcId) {
      return res.status(400).json({ error: "FDC ID parameter is required" });
    }

    const details = await usdaService.getFoodDetails(fdcId);

    res.json({
      success: true,
      data: details,
    });
  } catch (error) {
    console.error("Error getting food details:", error);
    res.status(500).json({ error: "Internal server error" });
  }
});

// ============================================================
// API: save ingredients detected by AI model
// ============================================================
router.get("/nutritionAPI/ingredients/detect", async (req, res) => {
  try {
    const { sessionId, ingredients } = req.query;

    if (!sessionId || !Array.isArray(ingredients)) {
      return res.status(400).json({ error: "invalid data" });
    }

    const savedIngredients = [];

    await prisma.$transaction(async (tx) => {
      for (const ingredient of ingredients) {
        let fdcId = null;
        const food = await usdaService.findAndCachedIngredient(
          tx,
          ingredient.name_en
        );
        fdcId = food ? food.fdcId : null;

        const saved = await tx.DetectedIngredient.create({
          data: {
            session_id: sessionId,
            name_vi: ingredient.name_vi,
            name_en: ingredient.name_en,
            quantity: ingredient.quantity || 0,
            freshness_level: ingredient.freshness_level || "fresh",
            freshness_score: ingredient.freshness_score || 100,
            is_usable: ingredient.is_usable !== false,
            usda_fdc_id: fdcId ? BigInt(fdcId) : null,
            image_url: ingredient.image_url || null,
          },
        });
      }
    });
  } catch (error) {
    console.error("Error saving detected ingredients:", error);
    res.status(500).json({ error: "Internal server error" });
  }
});
