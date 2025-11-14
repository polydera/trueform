import { RouteRecordSingleView } from "vue-router";
import CollisionExample from "./CollisionExample.vue";
import BooleanExample from "@/BooleanExample.vue";
import FormsIntersectionsExample from "@/FormsIntersectionsExample.vue";
import IsobandsExample from "@/IsobandsExample.vue";
import ScalarFieldIntersections from "@/ScalarFieldIntersections.vue";

interface ExampleRoute extends RouteRecordSingleView {
    label: string;
}

export const exampleRoutes: ExampleRoute[] = [
    { path: "/collision-example", component: CollisionExample, label: "Collision Example" },
    { path: "/boolean-example", component: BooleanExample, label: "Boolean Example" },
    { path: "/forms-intersections-example", component: FormsIntersectionsExample, label: "Forms Intersections Example" },
    { path: "/isobands-example", component: IsobandsExample, label: "Isobands Example" },
    { path: "/scalar-field-intersections", component: ScalarFieldIntersections, label: "Scalar Field Intersections" },
];
