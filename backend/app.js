import express from "express";
import cors from "cors";
import dotenv from "dotenv";
import chalk from "chalk";
import nutritionRoutes from "./src/nutrition/routes/nutrition.js";

// load config env
dotenv.config();

const app = express();
const PORT = process.env.PORT_BACKEND || 3000;

app.use(cors());
app.use(express.json({ limit: "50mb" }));
app.use(express.urlencoded({ extended: "true", limit: "50mb" }));

app.use("/api/nutrition", nutritionRoutes);

app.get("/health", (req, res) => {
  res.json({
    status: "OK",
    message: "Nutrition API is running smoothly",
    timestamp: new Date().toISOString(),
  });
});

// catch all error handler
app.use((err, req, res, next) => {
  res.status(err.status || 500).json({
    success: false,
    error: err.message || "Internal server error",
  });
});

const server = app.listen(PORT, () => {
  console.log(chalk.green(`Nutrition API running on port ${PORT}`));
  console.log(chalk.cyan(`USDA API integration enabled`));
  console.log(chalk.magenta(`PostgreSQL (Prisma) connected`));
});

export default app;
