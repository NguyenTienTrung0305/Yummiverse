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
            fdcId: foodData.fdcId,
          },

          update: {
            description: foodData.description,
            dataType: foodData.dataType,
            brandName: foodData.brandName || null,
            category: foodData.foodCategory || null,
            rawData: foodData,
            updatedAt: new Date()
          },

          create: {
            fdcId: foodData.fdcId,
            description: foodData.description,
            dataType: foodData.dataType,
            brandName: foodData.brandName || null,
            category: foodData.foodCategory || null,
            rawData: foodData
          }
        })

        const nutrition = this.parseNutrients(foodData.foodNutrients);


        await tx.nutritionCache.upsert({
          where: {
            usdaFdcId: usdaFood.fdcId,
          }

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
            vitaminA: nutrition.vitamin_a,
            vitaminC: nutrition.vitamin_c,
            vitaminD: nutrition.vitamin_d,
            calcium: nutrition.calcium,
            iron: nutrition.iron,
            potassium: nutrition.potassium
          },

          create: {
            usdaFdcId: foodData.fdcId,
            calories: nutrition.calories,
            protein: nutrition.protein,
            carbs: nutrition.carbs,
            fat: nutrition.fat,
            fiber: nutrition.fiber,
            sugars: nutrition.sugars,
            sodium: nutrition.sodium,
            cholesterol: nutrition.cholesterol,
            saturatedFat: nutrition.saturated_fat,
            vitaminA: nutrition.vitamin_a,
            vitaminC: nutrition.vitamin_c,
            vitaminD: nutrition.vitamin_d,
            calcium: nutrition.calcium,
            iron: nutrition.iron,
            potassium: nutrition.potassium
          }
        })

      });

    } catch (error) {
      console.error('Cache Error:', error.message);
      throw error;
    }
  }

  async getCachedFood(fdcId) {
    try {
      const food = await prisma.usdaFood.findUnique({
        where: {
          fdcId: fdcId,
        },
        include: {
          nutritionCache: true,
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
      'Energy': 'calories',
      'Protein': 'protein',
      'Carbohydrate, by difference': 'carbs',
      'Total lipid (fat)': 'fat',
      'Fiber, total dietary': 'fiber',
      'Sugars, total including NLEA': 'sugars',
      'Sodium, Na': 'sodium',
      'Cholesterol': 'cholesterol',
      'Fatty acids, total saturated': 'saturated_fat',
      'Vitamin A, RAE': 'vitamin_a',
      'Vitamin C, total ascorbic acid': 'vitamin_c',
      'Vitamin D (D2 + D3)': 'vitamin_d',
      'Calcium, Ca': 'calcium',
      'Iron, Fe': 'iron',
      'Potassium, K': 'potassium'
    }

    const result = {};
    
    foodNutrients.forEach(nutrient => {
      const name = nutrient.nutrient?.name;
      const key = nutrientMap[name];
      
      if (key) {
        let value = nutrient.amount || 0;
        result[key] = parseFloat(value.toFixed(2));
      }
    });

    return result;
  }


  async disconnect() {
    await prisma.$disconnect();
  }
}


module.exports = new USDAService();
