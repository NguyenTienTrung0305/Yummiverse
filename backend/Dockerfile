FROM node:18-alpine

WORKDIR /app

RUN apk add --no-cache bash

ENV NODE_ENV=development
ENV PORT=3000

COPY package*.json ./
RUN npm install


COPY prisma ./prisma/
RUN npx prisma generate

COPY . .

EXPOSE 3000

CMD ["npm", "start"]

