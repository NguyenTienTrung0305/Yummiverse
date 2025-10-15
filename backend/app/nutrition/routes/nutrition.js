import express from "express";
import usdaService from "../services/usdaService.js";
import { PrismaClient } from "@prisma/client";

const router = express.Router();
const prisma = new PrismaClient();

router.get("/search", async (req, res) => {
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

// ============================================================
// API: get details nutrition of food
// ============================================================
router.get("/details/:fdcId", async (req, res) => {
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
router.get("/ingredients/detect", async (req, res) => {
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

        savedIngredients.push(saved);
      }
    });

    res.json({
      success: true,
      message: "Sucessed sava ingredients",
      ingredients: savedIngredients,
    });
  } catch (error) {
    console.error("Error saving detected ingredients:", error);
    res.status(500).json({ error: "Internal server error" });
  }
});

// ============================================================
// API: get details nutrition of each ingredients detected in this session_id
// ============================================================
router.get("/ingredients/:session_id/nutrition", async (req, res) => {
  try {
    const { sessionId } = req.params;

    const detectIngredients = await prisma.DetectedIngredient.findMany({
      where: {
        session_id: sessionId
      },
      include: {
        food: {
          include: {
            nutrion: true,
          }
        }
      },
      orderBy: {
        detected_at: "desc"
      }
    })

    const ingredients = detectIngredients.map((item) => {
      const nutrion = item.food?.nutrition
      const quantity = parseFloat(item.quantity || 0)

      const isUsable = item.is_usable
      const nutritionForQuantity = (isUsable && nutrion) 
        ? {
          calories: ((nutrion.calories || 0) * quantity / 100).toFixed(2),
          protein: ((nutrition.protein || 0) * quantity / 100).toFixed(2),
          carbs: ((nutrition.carbs || 0) * quantity / 100).toFixed(2),
          fat: ((nutrition.fat || 0) * quantity / 100).toFixed(2),  
        }
        : null
        
      return {
        id: item.id,
        nameVi: item.name_vi,
        nameEn: item.name_en,
        quantity,
        freshness: {
          level: item.freshness_level,
          score: parseFloat(item.freshness_score || 0)
          usable: isUsable
        },
        nutrition: nutrition ? {
          per100g: {
            calories: parseFloat(nutrition.calories || 0),
            protein: parseFloat(nutrition.protein || 0),
            carbs: parseFloat(nutrition.carbs || 0),
            fat: parseFloat(nutrition.fat || 0),
            fiber: parseFloat(nutrition.fiber || 0),
            sugars: parseFloat(nutrition.sugars || 0),
            sodium: parseFloat(nutrition.sodium || 0),
            vitaminA: parseFloat(nutrition.vitamin_a || 0),
            vitaminC: parseFloat(nutrition.vitamin_c || 0),
            calcium: parseFloat(nutrition.calcium || 0),
            iron: parseFloat(nutrition.iron || 0),
          },
          forQuantity: nutritionForQuantity,
        } : null,
        usdaDescription: item.food?.description,
      }
    })

    res.json({
      success: true,
      sessionId: sessionId,
      ingredients: ingredients,
      summary: {
        totalIngredients: ingredients.length,
        totalUsable: ingredients.filter((item) => item.freshness.usable).length
        totalUnusable: ingredients.filter((item) => !item.freshness.usable).length
      }
    })

  } catch (error) {
    res.status(500).json({
      success: false,
      error: error.message,
    });
  }
});


router.get("/ingredients/:session_id/nutrition/summary", async (req, res) => {
  try {
    const { sessionId } = req.params

    const usableIngredients = await prisma.DetectedIngredient.findMany({
      where: {
        session_id: sessionId,
        is_usable: true, 
      },
      include: {
        food: {
          include: {
            nutrion: true
          }
        }
      }
    })

    const total = {
      calories: 0,
      protein: 0,
      carbs: 0,
      fat: 0,
    }

    usableIngredients.forEach((item) => {
      const nutrition_ = item.food?.nutrion
      const quantity_ = parseFloat(item.quantity || 0)

      if (!nutrition_) return
      
      total.calories += ((parseFloat(nutrition_.calories) || 0) * quantity_) / 100
      total.protein += ((parseFloat(nutrition_.protein) || 0) * quantity_) / 100
      total.carbs += ((parseFloat(nutrition_.carbs) || 0) * quantity_) / 100
      total.fat += ((parseFloat(nutrition_.fat) || 0) * quantity_) / 100
    })

    res.json({
      success: true,
      sessionId,
      totals: {
        calories: total.calories.toFixed(2),
        protein: total.protein.toFixed(2),
        carbs: total.carbs.toFixed(2),
        fat: total.fat.toFixed(2),
      },
      count: usableIngredients.length,
    });
  } catch (error) {
    console.error("Nutrition summary error:", error);
    res.status(500).json({ success: false, error: error.message });
  }
})


// ============================================================
// API: save user request in this session_id
// ============================================================
router.get("/requets", async(req, res) => {
  try {
    const { sessionId, userPrompt, cuisinePreference, dietaryRestrictions, allergies } = req.body

    const userRequest = await prisma.UserRequest.create({
      data: {
        session_id: sessionId,
        useruser_prompt: userPrompt,
        cuisine_preference: cuisinePreference,
        dietary_restrictions: dietaryRestrictions,
        allergies: allergies
      }
    })
  }
})
