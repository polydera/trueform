import './assets/main.css'
import { createRouter, createWebHistory } from "vue-router";
import { exampleRoutes } from "./Routes";

import { createApp } from 'vue'
import App from './App.vue'

const app = createApp(App)

const router = createRouter({
    history: createWebHistory(),
    routes: exampleRoutes,
});
app.use(router);

app.mount('#app')
