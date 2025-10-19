import axios from "axios";
import { PrismaClient } from "@prisma/client";

const prisma = new PrismaClient();

// USDA API configuration
const USDA_API_KEY = process.env.USDA_API_KEY;
const USDA_BASE_URL = "https://api.nal.usda.gov/fdc/v1";

class USDAService {
  async searchFoods(query, pageSize = 5) {
    try {
      const response = await axios.get(`${USDA_BASE_URL}/foods/search`, {
        params: {
          api_key: USDA_API_KEY,
          query: query,
          pageSize: pageSize,
          dataType: ["Foundation", "SR Legacy"],
        },
      });

      return response.data.foods || [];
    } catch (error) {
      console.error("Error searching foods:", error);
      throw error;
    }
  }

  async getFoodDetails(fdcId) {
    try {
      // get from cache first
      const cached = await this.getCachedFood(fdcId);
      if (cached) {
        return cached;
      }

      // call API if not in cache
      const response = await axios.get(`${USDA_BASE_URL}/food/${fdcId}`, {
        params: {
          api_key: USDA_API_KEY,
        },
      });

      const foodData = response.data;
      await this.cachedFoodData(foodData);
      return foodData;
    } catch (error) {
      console.error("USDA Details Error:", error.message);
      throw new Error("Can not get details info from USDA");
    }
  }

  async cachedFoodData(foodData) {
    try {
      await prisma.$transaction(async (tx) => {
        // Upsert = Insert or Update if exists
        const usdaFood = await tx.usdaFood.upsert({
          where: {
            fdc_id: foodData.fdcId,
          },

          update: {
            description: foodData.description,
            data_type: foodData.dataType,
            brand_name: foodData.brandName || null,
            category: foodData.foodCategory || null,
            raw_data: foodData,
            updated_at: new Date(),
          },

          create: {
            fdc_id: foodData.fdcId,
            description: foodData.description,
            data_type: foodData.dataType,
            brand_name: foodData.brandName || null,
            category: foodData.foodCategory || null,
            raw_data: foodData,
          },
        });

        const nutrition = this.parseNutrients(foodData.foodNutrients);

        await tx.nutritionCache.upsert({
          where: {
            usda_fdc_id: usdaFood.fdc_id,
          },

          update: {
            calories: nutrition.calories,
            protein: nutrition.protein,
            carbs: nutrition.carbs,
            fat: nutrition.fat,
            fiber: nutrition.fiber,
            sugars: nutrition.sugars,
            sodium: nutrition.sodium,
            cholesterol: nutrition.cholesterol,
            saturatedFat: nutrition.saturated_fat,
            vitamin_a: nutrition.vitamin_a,
            vitamin_c: nutrition.vitamin_c,
            vitamin_d: nutrition.vitamin_d,
            calcium: nutrition.calcium,
            iron: nutrition.iron,
            potassium: nutrition.potassium,
            updated_at: new Date(),
          },

          create: {
            usda_fdc_id: foodData.fdcId,
            calories: nutrition.calories,
            protein: nutrition.protein,
            carbs: nutrition.carbs,
            fat: nutrition.fat,
            fiber: nutrition.fiber,
            sugars: nutrition.sugars,
            sodium: nutrition.sodium,
            cholesterol: nutrition.cholesterol,
            saturatedFat: nutrition.saturated_fat,
            vitamin_a: nutrition.vitamin_a,
            vitamin_c: nutrition.vitamin_c,
            vitamin_d: nutrition.vitamin_d,
            calcium: nutrition.calcium,
            iron: nutrition.iron,
            potassium: nutrition.potassium,
          },
        });
      });
    } catch (error) {
      console.error("Cache Error:", error.message);
      throw error;
    }
  }

  async getCachedFood(fdcId) {
    try {
      const food = await prisma.usdaFood.findUnique({
        where: {
          fdc_id: fdcId,
        },
        include: {
          nutrition: true,
        },
      });
      return food;
    } catch (error) {
      console.error("Cache Lookup Error:", error.message);
      return null;
    }
  }

  parseNutrients(foodNutrients) {
    const nutrientMap = {
      Energy: "calories",
      Protein: "protein",
      "Carbohydrate, by difference": "carbs",
      "Total lipid (fat)": "fat",
      "Fiber, total dietary": "fiber",
      "Sugars, total including NLEA": "sugars",
      "Sodium, Na": "sodium",
      Cholesterol: "cholesterol",
      "Fatty acids, total saturated": "saturated_fat",
      "Vitamin A, RAE": "vitamin_a",
      "Vitamin C, total ascorbic acid": "vitamin_c",
      "Vitamin D (D2 + D3)": "vitamin_d",
      "Calcium, Ca": "calcium",
      "Iron, Fe": "iron",
      "Potassium, K": "potassium",
    };

    const result = {};

    foodNutrients.forEach((nutrient) => {
      const name = nutrient.nutrient?.name;
      const key = nutrientMap[name];

      if (key) {
        let value = nutrient.amount || 0;
        result[key] = parseFloat(value.toFixed(2));
      }
    });

    return result;
  }

  async findAndCachedIngredient(ingredientName) {
    try {
      const searchResults = await this.searchFoods(ingredientName, 5);
      if (searchResults.length === 0) {
        throw new Error("No matching food found in USDA");
      }

      const topResults = searchResults[0];
      const detailFood = await this.getFoodDetails(topResults.fdcId);

      return {
        fdcId: topResults.fdcId,
        description: topResults.description,
        nutrition: await this.getCachedFood(topResults.fdcId),
      };
    } catch (error) {
      console.error(
        `Error finding ingredient "${ingredientName}":`,
        error.message
      );
      throw error;
    }
  }

  async getNutritionForIngredients(ingredients) {
    const results = [];

    for (const ingredient of ingredients) {
      try {
        const data = await this.findAndCachedIngredient(ingredient);
        results.push({
          name: ingredient.name_vi,
          nameEn: ingredient.name_en,
          quantity: ingredient.quantity,
          freshness: ingredient.freshness_level,
          usable: ingredient.is_usable,
          fdcId: data.fdcId,
          nutrition: data.nutrition,
        });
      } catch (error) {
        console.error(
          `Error processing ingredient "${ingredient.name_vi}":`,
          error.message
        );
      }
    }

    return results;
  }

  async disconnect() {
    await prisma.$disconnect();
    return results;
  }

  async disconnect() {
    await prisma.$disconnect();
  }
}

export default new USDAService();
