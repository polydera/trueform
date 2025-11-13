import { RouteRecordSingleView } from "vue-router";
import CollisionExample from "./CollisionExample.vue";
import BooleanExample from "@/BooleanExample.vue";

interface ExampleRoute extends RouteRecordSingleView {
    label: string;
}

export const exampleRoutes: ExampleRoute[] = [
    { path: "/collision-example", component: CollisionExample, label: "Collision Example" },
    { path: "/boolean-example", component: BooleanExample, label: "Boolean Example" },
];
