-- USDA foods table
CREATE TABLE IF NOT EXISTS usda_foods (
  fdc_id BIGINT PRIMARY KEY,
  description TEXT,
  data_type VARCHAR(100),
  brand_name TEXT,
  category TEXT,
  raw_data JSONB NOT NULL,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Nutrition cache table
CREATE TABLE IF NOT EXISTS nutrition_cache (
  usda_fdc_id BIGINT PRIMARY KEY REFERENCES usda_foods(fdc_id) ON DELETE CASCADE,
  calories DECIMAL(10,2),
  protein DECIMAL(10,2),
  carbs DECIMAL(10,2),
  fat DECIMAL(10,2),
  fiber DECIMAL(10,2),
  sugars DECIMAL(10,2),
  sodium DECIMAL(10,2),
  cholesterol DECIMAL(10,2),
  saturated_fat DECIMAL(10,2),
  vitamin_a DECIMAL(10,2),
  vitamin_c DECIMAL(10,2),
  vitamin_d DECIMAL(10,2),
  calcium DECIMAL(10,2),
  iron DECIMAL(10,2),
  potassium DECIMAL(10,2),
  expiry_timestamp TIMESTAMP DEFAULT (CURRENT_TIMESTAMP + INTERVAL '24 hours'),
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);


-- Detected ingredients table
CREATE TABLE IF NOT EXISTS detected_ingredients (
  id SERIAL PRIMARY KEY,
  session_id VARCHAR(100) NOT NULL,
  name_vi TEXT NOT NULL,
  name_en TEXT,
  quantity DECIMAL(10,2),
  freshness_level VARCHAR(50) DEFAULT 'fresh',
  freshness_score DECIMAL(5,2) DEFAULT 100,
  is_usable BOOLEAN DEFAULT TRUE,
  usda_fdc_id BIGINT REFERENCES usda_foods(fdc_id),
  image_url TEXT,
  detected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);


-- User requests table
CREATE TABLE IF NOT EXISTS user_requests (
  id SERIAL PRIMARY KEY,
  session_id VARCHAR(100) NOT NULL,
  user_prompt TEXT,
  cuisine_preference VARCHAR(100),
  dietary_restrictions TEXT[],
  allergies TEXT[],
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Generated recipes table
CREATE TABLE IF NOT EXISTS generated_recipes (
  id SERIAL PRIMARY KEY,
  session_id VARCHAR(100) NOT NULL,
  user_request_id INTEGER REFERENCES user_requests(id) ON DELETE CASCADE,
  recipe_name VARCHAR(255) NOT NULL,
  cuisine_type VARCHAR(100),
  cooking_method VARCHAR(100),
  difficulty VARCHAR(50),
  cooking_time INTEGER,
  servings INTEGER,
  instructions TEXT,
  total_calories DECIMAL(10,2),
  total_protein DECIMAL(10,2),
  total_carbs DECIMAL(10,2),
  total_fat DECIMAL(10,2),
  llm_response JSONB,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Recipe ingredients table
CREATE TABLE IF NOT EXISTS recipe_ingredients (
  id SERIAL PRIMARY KEY,
  recipe_id INTEGER REFERENCES generated_recipes(id) ON DELETE CASCADE,
  detected_ingredient_id INTEGER REFERENCES detected_ingredients(id),
  quantity_used DECIMAL(10,2),
  unit VARCHAR(50),
  notes TEXT
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_usda_foods_description ON usda_foods USING gin(to_tsvector('english', description));
CREATE INDEX IF NOT EXISTS idx_detected_session ON detected_ingredients(session_id);
CREATE INDEX IF NOT EXISTS idx_user_requests_session ON user_requests(session_id);
CREATE INDEX IF NOT EXISTS idx_recipes_session ON generated_recipes(session_id);

-- Auto-update timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
  NEW.updated_at = CURRENT_TIMESTAMP;
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER update_usda_foods_updated_at BEFORE UPDATE ON usda_foods
  FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_nutrition_cache_updated_at BEFORE UPDATE ON nutrition_cache
  FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
